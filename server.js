import express from "express";
import cors from "cors";

const app = express();

// ============================================================
// V19
// Qwen3 + Granite + FLUX.2 Klein 4B
//
// Main improvements:
//   1. Model-aware context/output budgets
//   2. Dynamic max_tokens based on estimated input size
//   3. Client max_tokens / max_completion_tokens supported
//   4. Qwen numeric delta.content -> string normalization
//   5. Cherry Studio SSE compatibility
//   6. Function-calling fields preserved
//   7. FLUX routes kept separate
// ============================================================

// ============================================================
// Middleware
// ============================================================

app.use(cors());

app.use(
  express.json({
    limit: "50mb"
  })
);

app.use(
  express.urlencoded({
    extended: true,
    limit: "50mb"
  })
);

// ============================================================
// Environment
// ============================================================

const CF_ACCOUNT = (
  process.env.CF_ACCOUNT_ID || ""
).trim();

const CF_TOKEN = (
  process.env.CF_API_TOKEN || ""
).trim();

const PORT =
  process.env.PORT || 10000;

// ============================================================
// Models
// ============================================================

const MODELS = {
  QWEN: "@cf/qwen/qwen3-30b-a3b-fp8",
  GRANITE: "@cf/ibm-granite/granite-4.0-h-micro",
  IMAGE: "@cf/black-forest-labs/flux-2-klein-4b"
};

// ============================================================
// Model configuration
//
// IMPORTANT:
//
// contextWindow = official model context window
//
// defaultMaxTokens = used only when client gives no
// max_tokens / max_completion_tokens
//
// maxOutputTokens = our practical safety cap
//
// inputReserveTokens = safety margin for token estimation
//
// Qwen official context: 32,768
// Granite official context: 131,000
// ============================================================

const MODEL_CONFIG = {
  [MODELS.QWEN]: {
    name: "Qwen3 30B A3B FP8",

    contextWindow: 32768,

    defaultMaxTokens: 8192,

    maxOutputTokens: 16384,

    inputReserveTokens: 1024
  },

  [MODELS.GRANITE]: {
    name: "Granite 4.0 H Micro",

    contextWindow: 131000,

    defaultMaxTokens: 16384,

    maxOutputTokens: 32768,

    inputReserveTokens: 1024
  }
};

// ============================================================
// Token estimation
//
// We intentionally do NOT pretend this is an exact tokenizer.
//
// Because the proxy does not load the actual model tokenizer,
// we use a conservative character-based estimate.
//
// Approximation:
//   estimated tokens ~= chars / 2.5
//
// This is intentionally conservative for:
//   Chinese
//   source code
//   JSON
//   mixed-language prompts
//
// The goal is not exact billing.
// The goal is preventing:
//   input + output > context window
// ============================================================

const CHARS_PER_TOKEN_ESTIMATE = 2.5;

function estimateInputTokens(messages) {
  if (!Array.isArray(messages)) {
    return 0;
  }

  const totalChars =
    messages.reduce(
      (sum, message) =>
        sum +
        String(
          message?.content || ""
        ).length,
      0
    );

  // Small overhead for role/message boundaries
  const messageOverhead =
    messages.length * 8;

  const estimated =
    Math.ceil(
      totalChars /
        CHARS_PER_TOKEN_ESTIMATE
    ) + messageOverhead;

  return estimated;
}

// ============================================================
// Environment warnings
// ============================================================

if (!CF_ACCOUNT) {
  console.warn(
    "WARNING: CF_ACCOUNT_ID is not configured"
  );
}

if (!CF_TOKEN) {
  console.warn(
    "WARNING: CF_API_TOKEN is not configured"
  );
}

// ============================================================
// Basic endpoints
// ============================================================

app.get("/ping", (req, res) => {
  res
    .type("text/plain")
    .send(
      `pong - ${MODELS.QWEN} + ${MODELS.GRANITE} + ${MODELS.IMAGE} alive - ${new Date().toISOString()}`
    );
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    models: Object.values(MODELS),
    time: new Date().toISOString()
  });
});

app.get("/", (req, res) => {
  res
    .type("text/plain")
    .send(
      "ready V19 qwen+granite+flux"
    );
});

// ============================================================
// OpenAI-compatible /v1/models
// ============================================================

app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",

    data: [
      {
        id: MODELS.QWEN,
        object: "model",
        owned_by: "qwen"
      },
      {
        id: MODELS.GRANITE,
        object: "model",
        owned_by: "ibm"
      },
      {
        id: MODELS.IMAGE,
        object: "model",
        owned_by: "black-forest"
      }
    ]
  });
});

// ============================================================
// Resolve chat model
// ============================================================

function resolveChatModel(inputModel) {
  let model = String(
    inputModel || MODELS.QWEN
  ).trim();

  // Existing compatibility
  if (
    model.includes("llama") ||
    model.includes("fast")
  ) {
    model = MODELS.QWEN;
  }

  if (
    model.includes("ibm/granite") ||
    model.includes("granite-4.0") ||
    model.includes("granite")
  ) {
    model = MODELS.GRANITE;
  }

  // Only Qwen / Granite allowed
  if (
    !model.includes("qwen") &&
    !model.includes("granite")
  ) {
    model = MODELS.QWEN;
  }

  return model;
}

// ============================================================
// Resolve requested max_tokens
//
// Priority:
//
//   1. max_completion_tokens
//   2. max_tokens
//   3. null = no client value
//
// We intentionally return null when the client didn't specify
// anything, because dynamic calculation happens later.
// ============================================================

function getClientRequestedMaxTokens(
  reqBody
) {
  let rawValue = null;

  if (
    reqBody?.max_completion_tokens !==
      undefined &&
    reqBody?.max_completion_tokens !==
      null
  ) {
    rawValue =
      reqBody.max_completion_tokens;
  } else if (
    reqBody?.max_tokens !==
      undefined &&
    reqBody?.max_tokens !==
      null
  ) {
    rawValue =
      reqBody.max_tokens;
  }

  if (rawValue === null) {
    return null;
  }

  const numberValue =
    Number(rawValue);

  if (
    !Number.isFinite(
      numberValue
    ) ||
    numberValue <= 0
  ) {
    console.warn(
      "Invalid client max_tokens:",
      rawValue
    );

    return null;
  }

  return Math.floor(
    numberValue
  );
}

// ============================================================
// Calculate safe max_tokens
//
// Rules:
//
// 1. Client value is honored when possible.
// 2. Never exceed our model safety cap.
// 3. Never exceed remaining estimated context.
// 4. If no client value, use model default.
// 5. If remaining context is too small, return error info.
//
// IMPORTANT:
// This is an estimate, not an exact tokenizer calculation.
// ============================================================

function calculateMaxTokens(
  model,
  messages,
  reqBody
) {
  const config =
    MODEL_CONFIG[model];

  if (!config) {
    return {
      maxTokens: 4096,
      estimatedInputTokens: 0,
      availableOutputTokens: 4096,
      clientRequested: null,
      error: null
    };
  }

  const estimatedInputTokens =
    estimateInputTokens(
      messages
    );

  const availableOutputTokens =
    Math.max(
      0,
      config.contextWindow -
        estimatedInputTokens -
        config.inputReserveTokens
    );

  const clientRequested =
    getClientRequestedMaxTokens(
      reqBody
    );

  let requested;

  if (
    clientRequested !== null
  ) {
    requested =
      clientRequested;
  } else {
    requested =
      config.defaultMaxTokens;
  }

  // Never exceed practical safety cap
  requested =
    Math.min(
      requested,
      config.maxOutputTokens
    );

  // Never exceed remaining context
  requested =
    Math.min(
      requested,
      availableOutputTokens
    );

  // A very small remaining output space is not useful
  if (
    requested < 1
  ) {
    return {
      maxTokens: 0,

      estimatedInputTokens,

      availableOutputTokens,

      clientRequested,

      error:
        `Input is too large for ${config.name}. ` +
        `Estimated input: ${estimatedInputTokens} tokens, ` +
        `context window: ${config.contextWindow} tokens.`
    };
  }

  return {
    maxTokens:
      Math.floor(requested),

    estimatedInputTokens,

    availableOutputTokens,

    clientRequested,

    error: null
  };
}

// ============================================================
// Normalize input messages
// ============================================================

function normalizeMessages(
  inputMessages
) {
  if (
    !Array.isArray(
      inputMessages
    )
  ) {
    return [];
  }

  return inputMessages
    .map((message) => {
      let content =
        message?.content;

      // --------------------------------------------------------
      // OpenAI content array
      // --------------------------------------------------------

      if (
        Array.isArray(
          content
        )
      ) {
        content =
          content
            .map(
              (item) => {
                if (
                  typeof item ===
                  "string"
                ) {
                  return item;
                }

                if (
                  item?.text !==
                    undefined &&
                  item?.text !==
                    null
                ) {
                  return String(
                    item.text
                  );
                }

                if (
                  item?.content !==
                    undefined &&
                  item?.content !==
                    null
                ) {
                  return String(
                    item.content
                  );
                }

                return "";
              }
            )
            .join("\n");
      }

      return {
        role: String(
          message?.role ||
            "user"
        ),

        content: String(
          content ?? ""
        )
      };
    })

    .filter(
      (message) =>
        message.content
          .length > 0
    );
}

// ============================================================
// Normalize Cloudflare response object
//
// PRIMARY FIX:
//
// Qwen may return:
//
//   delta.content = 1
//
// Cherry Studio expects:
//
//   delta.content = "1"
//
// We convert non-string primitive values to strings.
//
// We preserve:
//   null
//   usage
//   reasoning_content
//   token_ids
//   tool_calls
//   finish_reason
//   model
//   id
//   created
// ============================================================

function normalizeChatResponseObject(
  obj
) {
  if (
    !obj ||
    typeof obj !==
      "object"
  ) {
    return obj;
  }

  // ----------------------------------------------------------
  // choices
  // ----------------------------------------------------------

  if (
    Array.isArray(
      obj.choices
    )
  ) {
    obj.choices =
      obj.choices.map(
        (choice) => {
          if (
            !choice ||
            typeof choice !==
              "object"
          ) {
            return choice;
          }

          // --------------------------------------------------
          // delta
          // --------------------------------------------------

          if (
            choice.delta &&
            typeof choice.delta ===
              "object"
          ) {
            const delta =
              choice.delta;

            // Qwen numeric content fix
            if (
              delta.content !==
                undefined &&
              delta.content !==
                null &&
              typeof delta.content !==
                "string"
            ) {
              delta.content =
                String(
                  delta.content
                );
            }

            // reasoning_content
            if (
              delta.reasoning_content !==
                undefined &&
              delta.reasoning_content !==
                null &&
              typeof delta.reasoning_content !==
                "string"
            ) {
              delta.reasoning_content =
                String(
                  delta.reasoning_content
                );
            }
          }

          // --------------------------------------------------
          // message
          // --------------------------------------------------

          if (
            choice.message &&
            typeof choice.message ===
              "object"
          ) {
            const message =
              choice.message;

            if (
              message.content !==
                undefined &&
              message.content !==
                null &&
              typeof message.content !==
                "string"
            ) {
              message.content =
                String(
                  message.content
                );
            }

            if (
              message.reasoning_content !==
                undefined &&
              message.reasoning_content !==
                null &&
              typeof message.reasoning_content !==
                "string"
            ) {
              message.reasoning_content =
                String(
                  message.reasoning_content
                );
            }
          }

          return choice;
        }
      );
  }

  return obj;
}

// ============================================================
// Build Chat payload
// ============================================================

function buildChatPayload(
  reqBody,
  model,
  messages,
  isStream
) {
  const tokenInfo =
    calculateMaxTokens(
      model,
      messages,
      reqBody
    );

  if (
    tokenInfo.error
  ) {
    return {
      payload: null,
      tokenInfo
    };
  }

  const payload = {
    model,
    messages,
    stream: isStream,
    max_tokens:
      tokenInfo.maxTokens
  };

  // ----------------------------------------------------------
  // Generation parameters
  // ----------------------------------------------------------

  const forwardParams = [
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "response_format",
    "stream_options",
    "raw"
  ];

  for (
    const key of
      forwardParams
  ) {
    if (
      reqBody?.[key] !==
      undefined
    ) {
      payload[key] =
        reqBody[key];
    }
  }

  // ----------------------------------------------------------
  // Function calling / agent support
  //
  // Useful for coding agents and VS Code integrations.
  // Only copied when the client actually sends them.
  // ----------------------------------------------------------

  const toolParams = [
    "tools",
    "tool_choice",
    "parallel_tool_calls"
  ];

  for (
    const key of
      toolParams
  ) {
    if (
      reqBody?.[key] !==
      undefined
    ) {
      payload[key] =
        reqBody[key];
    }
  }

  return {
    payload,
    tokenInfo
  };
}

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post(
  "/v1/chat/completions",
  async (req, res) => {

    const isStream =
      req.body?.stream ===
      true;

    // --------------------------------------------------------
    // Resolve model
    // --------------------------------------------------------

    const model =
      resolveChatModel(
        req.body?.model
      );

    // --------------------------------------------------------
    // Normalize messages
    // --------------------------------------------------------

    const messages =
      normalizeMessages(
        req.body?.messages
      );

    // --------------------------------------------------------
    // Build payload
    // --------------------------------------------------------

    const {
      payload,
      tokenInfo
    } =
      buildChatPayload(
        req.body,
        model,
        messages,
        isStream
      );

    // --------------------------------------------------------
    // Log incoming request
    // --------------------------------------------------------

    console.log(
      "=================================================="
    );

    console.log(
      "CHAT REQUEST"
    );

    console.log(
      "Model:",
      model
    );

    console.log(
      "Stream:",
      isStream
    );

    console.log(
      "RAW max_tokens:",
      req.body?.max_tokens
    );

    console.log(
      "RAW max_completion_tokens:",
      req.body?.max_completion_tokens
    );

    console.log(
      "Estimated input tokens:",
      tokenInfo.estimatedInputTokens
    );

    console.log(
      "Available output tokens:",
      tokenInfo.availableOutputTokens
    );

    console.log(
      "Client requested max_tokens:",
      tokenInfo.clientRequested
    );

    console.log(
      "Resolved max_tokens:",
      tokenInfo.maxTokens
    );

    console.log(
      "Message count:",
      messages.length
    );

    console.log(
      "Input chars:",
      messages.reduce(
        (total, message) =>
          total +
          message.content
            .length,
        0
      )
    );

    console.log(
      "stream_options:",
      req.body?.stream_options
    );

    // --------------------------------------------------------
    // Payload error
    // --------------------------------------------------------

    if (
      tokenInfo.error
    ) {
      console.error(
        "CONTEXT ERROR:",
        tokenInfo.error
      );

      return res
        .status(400)
        .json({
          error: {
            message:
              tokenInfo.error
          }
        });
    }

    // --------------------------------------------------------
    // Final payload log
    // --------------------------------------------------------

    console.log(
      "FINAL CLOUDFLARE PAYLOAD:",
      JSON.stringify({
        model:
          payload.model,

        stream:
          payload.stream,

        max_tokens:
          payload.max_tokens,

        temperature:
          payload.temperature,

        top_p:
          payload.top_p,

        top_k:
          payload.top_k,

        stream_options:
          payload.stream_options,

        has_tools:
          Array.isArray(
            payload.tools
          )
      })
    );

    console.log(
      "=================================================="
    );

    // --------------------------------------------------------
    // Cloudflare endpoint
    // --------------------------------------------------------

    const cfUrl =
      `https://api.cloudflare.com/client/v4/accounts/` +
      `${CF_ACCOUNT}/ai/v1/chat/completions`;

    // --------------------------------------------------------
    // AbortController
    // --------------------------------------------------------

    const controller =
      new AbortController();

    res.on(
      "close",
      () => {
        if (
          !res.writableEnded
        ) {
          console.warn(
            "CLIENT DISCONNECTED -> aborting Cloudflare request"
          );

          controller.abort();
        }
      }
    );

    try {

      // ======================================================
      // STREAMING
      // ======================================================

      if (isStream) {

        // ----------------------------------------------------
        // SSE headers
        // ----------------------------------------------------

        res.status(200);

        res.setHeader(
          "Content-Type",
          "text/event-stream; charset=utf-8"
        );

        res.setHeader(
          "Cache-Control",
          "no-cache, no-transform"
        );

        res.setHeader(
          "Connection",
          "keep-alive"
        );

        res.setHeader(
          "X-Accel-Buffering",
          "no"
        );

        if (
          typeof res.flushHeaders ===
          "function"
        ) {
          res.flushHeaders();
        }

        // ----------------------------------------------------
        // Call Cloudflare
        // ----------------------------------------------------

        const cfRes =
          await fetch(
            cfUrl,
            {
              method: "POST",

              headers: {
                Authorization:
                  `Bearer ${CF_TOKEN}`,

                "Content-Type":
                  "application/json",

                Accept:
                  "text/event-stream"
              },

              body:
                JSON.stringify(
                  payload
                ),

              signal:
                controller.signal
            }
          );

        console.log(
          "CF STREAM STATUS:",
          cfRes.status
        );

        console.log(
          "CF STREAM CONTENT-TYPE:",
          cfRes.headers.get(
            "content-type"
          )
        );

        // ----------------------------------------------------
        // Cloudflare HTTP error
        // ----------------------------------------------------

        if (
          !cfRes.ok
        ) {
          const errorText =
            await cfRes.text();

          console.error(
            "CF STREAM ERROR:",
            errorText
          );

          if (
            !res.writableEnded
          ) {

            res.write(
              `data: ${JSON.stringify(
                {
                  error: {
                    message:
                      errorText,
                    status:
                      cfRes.status
                  }
                }
              )}\n\n`
            );

            res.write(
              "data: [DONE]\n\n"
            );

            res.end();
          }

          return;
        }

        // ----------------------------------------------------
        // No upstream body
        // ----------------------------------------------------

        if (
          !cfRes.body
        ) {
          console.error(
            "CF STREAM ERROR: empty response body"
          );

          if (
            !res.writableEnded
          ) {

            res.write(
              `data: ${JSON.stringify(
                {
                  error: {
                    message:
                      "Cloudflare returned an empty streaming body"
                  }
                }
              )}\n\n`
            );

            res.write(
              "data: [DONE]\n\n"
            );

            res.end();
          }

          return;
        }

        // ----------------------------------------------------
        // SSE parser / compatibility layer
        // ----------------------------------------------------

        const reader =
          cfRes.body.getReader();

        const decoder =
          new TextDecoder();

        let buffer = "";

        let receivedDone =
          false;

        while (true) {

          const {
            done,
            value
          } =
            await reader.read();

          if (
            done
          ) {
            break;
          }

          if (!value) {
            continue;
          }

          // --------------------------------------------------
          // Decode incoming bytes
          // --------------------------------------------------

          buffer +=
            decoder.decode(
              value,
              {
                stream: true
              }
            );

          const lines =
            buffer.split(
              /\r?\n/
            );

          // --------------------------------------------------
          // Keep incomplete final line
          // --------------------------------------------------

          buffer =
            lines.pop() ||
            "";

          // --------------------------------------------------
          // Process complete lines
          // --------------------------------------------------

          for (
            const line of
              lines
          ) {

            // Empty separator
            if (
              line.trim() ===
              ""
            ) {
              continue;
            }

            // ------------------------------------------------
            // SSE comments
            // ------------------------------------------------

            if (
              line.startsWith(
                ":"
              )
            ) {
              continue;
            }

            // ------------------------------------------------
            // Only data lines
            // ------------------------------------------------

            if (
              !line.startsWith(
                "data:"
              )
            ) {
              continue;
            }

            const data =
              line
                .slice(5)
                .trim();

            // ------------------------------------------------
            // DONE
            // ------------------------------------------------

            if (
              data ===
              "[DONE]"
            ) {

              receivedDone =
                true;

              if (
                !res.writableEnded
              ) {
                res.write(
                  "data: [DONE]\n\n"
                );
              }

              continue;
            }

            if (!data) {
              continue;
            }

            // ------------------------------------------------
            // Parse JSON
            // ------------------------------------------------

            let chunk;

            try {

              chunk =
                JSON.parse(
                  data
                );

            } catch (error) {

              console.warn(
                "Skipping malformed SSE JSON:",
                data
              );

              continue;
            }

            // ------------------------------------------------
            // Normalize provider-specific type differences
            // ------------------------------------------------

            chunk =
              normalizeChatResponseObject(
                chunk
              );

            // ------------------------------------------------
            // Send normalized SSE
            // ------------------------------------------------

            if (
              !res.writableEnded
            ) {

              res.write(
                `data: ${JSON.stringify(
                  chunk
                )}\n\n`
              );
            }
          }
        }

        // ----------------------------------------------------
        // Flush decoder
        // ----------------------------------------------------

        buffer +=
          decoder.decode();

        // ----------------------------------------------------
        // Process remaining line
        // ----------------------------------------------------

        if (
          buffer.trim()
        ) {

          const remainingLines =
            buffer.split(
              /\r?\n/
            );

          for (
            const line of
              remainingLines
          ) {

            if (
              !line.startsWith(
                "data:"
              )
            ) {
              continue;
            }

            const data =
              line
                .slice(5)
                .trim();

            if (
              data ===
              "[DONE]"
            ) {

              receivedDone =
                true;

              if (
                !res.writableEnded
              ) {
                res.write(
                  "data: [DONE]\n\n"
                );
              }

              continue;
            }

            if (!data) {
              continue;
            }

            try {

              let chunk =
                JSON.parse(
                  data
                );

              chunk =
                normalizeChatResponseObject(
                  chunk
                );

              if (
                !res.writableEnded
              ) {
                res.write(
                  `data: ${JSON.stringify(
                    chunk
                  )}\n\n`
                );
              }

            } catch (error) {

              console.warn(
                "Skipping final malformed SSE JSON:",
                data
              );
            }
          }
        }

        // ----------------------------------------------------
        // Ensure DONE
        // ----------------------------------------------------

        if (
          !receivedDone &&
          !res.writableEnded
        ) {

          console.warn(
            "WARNING: Cloudflare stream ended without [DONE]"
          );

          res.write(
            "data: [DONE]\n\n"
          );
        }

        if (
          !res.writableEnded
        ) {
          res.end();
        }

        console.log(
          "CF STREAM ENDED. DONE RECEIVED:",
          receivedDone
        );

        return;
      }

      // ======================================================
      // NON-STREAMING
      // ======================================================

      const cfRes =
        await fetch(
          cfUrl,
          {
            method: "POST",

            headers: {
              Authorization:
                `Bearer ${CF_TOKEN}`,

              "Content-Type":
                "application/json",

              Accept:
                "application/json"
            },

            body:
              JSON.stringify(
                payload
              ),

            signal:
              controller.signal
          }
        );

      console.log(
        "CF NON-STREAM STATUS:",
        cfRes.status
      );

      const text =
        await cfRes.text();

      // ------------------------------------------------------
      // Cloudflare error
      // ------------------------------------------------------

      if (
        !cfRes.ok
      ) {

        console.error(
          "CF ERROR:",
          text
        );

        return res
          .status(
            cfRes.status
          )
          .json({
            error: {
              message:
                text
            }
          });
      }

      // ------------------------------------------------------
      // Parse / normalize JSON response
      // ------------------------------------------------------

      try {

        let data =
          JSON.parse(
            text
          );

        data =
          normalizeChatResponseObject(
            data
          );

        res.setHeader(
          "Content-Type",
          "application/json"
        );

        return res.json(
          data
        );

      } catch (error) {

        console.warn(
          "Cloudflare returned non-JSON response:",
          text
        );

        res.setHeader(
          "Content-Type",
          "application/json"
        );

        return res.send(
          text
        );
      }

    } catch (error) {

      // ======================================================
      // General error
      // ======================================================

      console.error(
        "CHAT ERROR:",
        error
      );

      // ------------------------------------------------------
      // Client disconnected
      // ------------------------------------------------------

      if (
        error?.name ===
        "AbortError"
      ) {

        console.warn(
          "Cloudflare request aborted because client disconnected."
        );

        return;
      }

      // ------------------------------------------------------
      // Streaming error
      // ------------------------------------------------------

      if (
        isStream
      ) {

        if (
          !res.headersSent
        ) {

          return res
            .status(500)
            .json({
              error: {
                message:
                  error?.message ||
                  "Streaming error"
              }
            });
        }

        if (
          !res.writableEnded
        ) {

          try {

            res.write(
              `data: ${JSON.stringify(
                {
                  error: {
                    message:
                      error?.message ||
                      "Streaming error"
                  }
                }
              )}\n\n`
            );

            res.write(
              "data: [DONE]\n\n"
            );

            res.end();

          } catch (_) {
            // Client already disconnected
          }
        }

        return;
      }

      // ------------------------------------------------------
      // Normal error
      // ------------------------------------------------------

      if (
        !res.headersSent
      ) {

        return res
          .status(500)
          .json({
            error: {
              message:
                error?.message ||
                "Request failed"
            }
          });
      }
    }
  }
);

// ============================================================
// FLUX.2 Klein 4B IMAGE HANDLER
//
// Kept separate from Qwen / Granite chat route.
// ============================================================

async function handleImage(
  req,
  res
) {

  try {

    const model =
      MODELS.IMAGE;

    const prompt =
      String(
        req.body?.prompt ||
        "full body photo of the same person, same face"
      );

    const size =
      String(
        req.body?.size ||
        "910x512"
      ).split("x");

    const width =
      String(
        parseInt(
          size[0],
          10
        ) || 910
      );

    const height =
      String(
        parseInt(
          size[1],
          10
        ) || 512
      );

    const imgInput =
      req.body?.image ||
      req.body?.image_b64;

    // --------------------------------------------------------
    // Multipart form
    // --------------------------------------------------------

    const form =
      new FormData();

    form.append(
      "prompt",
      prompt
    );

    form.append(
      "width",
      width
    );

    form.append(
      "height",
      height
    );

    // Keep your existing 4-step FLUX configuration
    form.append(
      "steps",
      "4"
    );

    // --------------------------------------------------------
    // IMG2IMG
    // --------------------------------------------------------

    if (
      imgInput
    ) {

      const inputString =
        String(
          imgInput
        );

      const b64 =
        inputString.includes(
          ","
        )
          ? inputString
              .split(",")[1]
          : inputString;

      const buffer =
        Buffer.from(
          b64,
          "base64"
        );

      form.append(
        "image",

        new Blob(
          [buffer],
          {
            type:
              "image/jpeg"
          }
        ),

        "input.jpg"
      );

      form.append(
        "strength",

        String(
          req.body?.strength ??
          0.5
        )
      );

      console.log(
        "KLEIN IMG2IMG MULTIPART, size:",
        buffer.length
      );

    } else {

      console.log(
        "KLEIN TEXT2IMG MULTIPART"
      );
    }

    // --------------------------------------------------------
    // Cloudflare image API
    // --------------------------------------------------------

    const cfUrl =
      `https://api.cloudflare.com/client/v4/accounts/` +
      `${CF_ACCOUNT}/ai/run/${model}`;

    const cfRes =
      await fetch(
        cfUrl,
        {
          method: "POST",

          headers: {
            Authorization:
              `Bearer ${CF_TOKEN}`

            // Do NOT manually set Content-Type.
            // fetch() creates multipart boundary.
          },

          body:
            form
        }
      );

    const text =
      await cfRes.text();

    console.log(
      "CF IMAGE STATUS:",
      cfRes.status
    );

    // --------------------------------------------------------
    // Cloudflare image error
    // --------------------------------------------------------

    if (
      !cfRes.ok
    ) {

      console.error(
        "CF KLEIN ERROR:",
        text
      );

      return res
        .status(
          cfRes.status
        )
        .json({
          error: {
            message:
              text
          }
        });
    }

    // --------------------------------------------------------
    // Parse image JSON
    // --------------------------------------------------------

    let data;

    try {

      data =
        JSON.parse(
          text
        );

    } catch (error) {

      console.error(
        "FLUX JSON PARSE ERROR:",
        text
      );

      return res
        .status(502)
        .json({
          error: {
            message:
              "Cloudflare returned invalid JSON"
          }
        });
    }

    // --------------------------------------------------------
    // OpenAI-compatible image response
    // --------------------------------------------------------

    return res.json({
      created:
        Date.now(),

      data: [
        {
          b64_json:
            data?.result?.image ??
            data?.result
        }
      ]
    });

  } catch (error) {

    console.error(
      "FINAL IMAGE ERROR:",
      error
    );

    if (
      !res.headersSent
    ) {

      return res
        .status(500)
        .json({
          error: {
            message:
              error?.message ||
              "Image generation failed"
          }
        });
    }
  }
}

// ============================================================
// Image endpoints
// ============================================================

app.post(
  "/v1/images/generations",
  handleImage
);

app.post(
  "/v1/images/edits",
  handleImage
);

// ============================================================
// Express JSON parser error handler
//
// Converts default HTML:
//
//   <pre>Bad Request</pre>
//
// into JSON.
// ============================================================

app.use(
  (
    error,
    req,
    res,
    next
  ) => {

    if (
      error instanceof
        SyntaxError &&
      error?.status ===
        400 &&
      "body" in error
    ) {

      console.error(
        "INVALID JSON BODY:",
        error.message
      );

      return res
        .status(400)
        .json({
          error: {
            message:
              "Invalid JSON request body",

            detail:
              error.message
          }
        });
    }

    console.error(
      "UNHANDLED EXPRESS ERROR:",
      error
    );

    if (
      res.headersSent
    ) {
      return next(
        error
      );
    }

    return res
      .status(
        error?.status ||
          500
      )
      .json({
        error: {
          message:
            error?.message ||
            "Internal server error"
        }
      });
  }
);

// ============================================================
// Start server
// ============================================================

app.listen(
  PORT,
  () => {

    console.log(
      "=================================================="
    );

    console.log(
      "V19 running on port",
      PORT
    );

    console.log(
      "QWEN:",
      MODELS.QWEN
    );

    console.log(
      "GRANITE:",
      MODELS.GRANITE
    );

    console.log(
      "IMAGE:",
      MODELS.IMAGE
    );

    console.log(
      "QWEN context:",
      MODEL_CONFIG[
        MODELS.QWEN
      ].contextWindow
    );

    console.log(
      "QWEN default output:",
      MODEL_CONFIG[
        MODELS.QWEN
      ].defaultMaxTokens
    );

    console.log(
      "QWEN max output:",
      MODEL_CONFIG[
        MODELS.QWEN
      ].maxOutputTokens
    );

    console.log(
      "GRANITE context:",
      MODEL_CONFIG[
        MODELS.GRANITE
      ].contextWindow
    );

    console.log(
      "GRANITE default output:",
      MODEL_CONFIG[
        MODELS.GRANITE
      ].defaultMaxTokens
    );

    console.log(
      "GRANITE max output:",
      MODEL_CONFIG[
        MODELS.GRANITE
      ].maxOutputTokens
    );

    console.log(
      "=================================================="
    );
  }
);


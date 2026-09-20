import express from "express";
import cors from "cors";

const app = express();

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

const CF_ACCOUNT = (process.env.CF_ACCOUNT_ID || "").trim();
const CF_TOKEN = (process.env.CF_API_TOKEN || "").trim();
const PORT = process.env.PORT || 10000;

// ============================================================
// Models
// ============================================================

const MODELS = {
  QWEN: "@cf/qwen/qwen3-30b-a3b-fp8",
  GRANITE: "@cf/ibm-granite/granite-4.0-h-micro",
  IMAGE: "@cf/black-forest-labs/flux-2-klein-4b"
};

// ============================================================
// Default max_tokens
//
// IMPORTANT:
// Only used when the client does NOT provide
// max_tokens / max_completion_tokens.
//
// Client supplied value always wins.
// ============================================================

const DEFAULT_MAX_TOKENS = {
  QWEN: 4096,
  GRANITE: 4096
};

// ============================================================
// Environment warnings
// ============================================================

if (!CF_ACCOUNT) {
  console.warn("WARNING: CF_ACCOUNT_ID is not configured");
}

if (!CF_TOKEN) {
  console.warn("WARNING: CF_API_TOKEN is not configured");
}

// ============================================================
// Basic routes
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
    .send("ready V18 qwen+granite+flux");
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

  // Existing compatibility rules
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

  // Only Qwen / Granite allowed on chat endpoint
  if (
    !model.includes("qwen") &&
    !model.includes("granite")
  ) {
    model = MODELS.QWEN;
  }

  return model;
}

// ============================================================
// Resolve max_tokens
//
// Priority:
//
// 1. max_completion_tokens
// 2. max_tokens
// 3. model-specific default
//
// IMPORTANT:
// Never overwrite a valid client supplied value.
// ============================================================

function resolveMaxTokens(reqBody, model) {
  let rawValue = null;

  if (
    reqBody?.max_completion_tokens !== undefined &&
    reqBody?.max_completion_tokens !== null
  ) {
    rawValue =
      reqBody.max_completion_tokens;
  } else if (
    reqBody?.max_tokens !== undefined &&
    reqBody?.max_tokens !== null
  ) {
    rawValue =
      reqBody.max_tokens;
  }

  // ----------------------------------------------------------
  // Client explicitly supplied max tokens
  // ----------------------------------------------------------

  if (rawValue !== null) {
    const n = Number(rawValue);

    if (
      Number.isFinite(n) &&
      n > 0
    ) {
      return Math.floor(n);
    }

    console.warn(
      "Invalid max_tokens received:",
      rawValue
    );
  }

  // ----------------------------------------------------------
  // No valid client value -> model default
  // ----------------------------------------------------------

  if (model.includes("granite")) {
    return DEFAULT_MAX_TOKENS.GRANITE;
  }

  if (model.includes("qwen")) {
    return DEFAULT_MAX_TOKENS.QWEN;
  }

  return 4096;
}

// ============================================================
// Normalize incoming messages
//
// Handles:
//
// content: "hello"
//
// or:
//
// content: [
//   { type: "text", text: "hello" }
// ]
// ============================================================

function normalizeMessages(inputMessages) {
  if (!Array.isArray(inputMessages)) {
    return [];
  }

  return inputMessages
    .map((m) => {
      let content = m?.content;

      if (Array.isArray(content)) {
        content = content
          .map((item) => {
            if (typeof item === "string") {
              return item;
            }

            if (
              item?.text !== undefined &&
              item?.text !== null
            ) {
              return String(item.text);
            }

            if (
              item?.content !== undefined &&
              item?.content !== null
            ) {
              return String(item.content);
            }

            return "";
          })
          .join("\n");
      }

      return {
        role: String(
          m?.role || "user"
        ),
        content: String(
          content ?? ""
        )
      };
    })
    .filter(
      (m) =>
        m.content.length > 0
    );
}

// ============================================================
// Normalize Cloudflare/OpenAI-compatible response
//
// IMPORTANT QWEN FIX:
//
// Cloudflare Qwen can return:
//
// {
//   "delta": {
//      "content": 1
//   }
// }
//
// Cherry Studio expects:
//
// {
//   "delta": {
//      "content": "1"
//   }
// }
//
// We normalize non-string scalar values to strings.
//
// We DO NOT remove usage, reasoning_content, token_ids, etc.
// ============================================================

function normalizeChatResponseObject(obj) {
  if (
    !obj ||
    typeof obj !== "object"
  ) {
    return obj;
  }

  // ----------------------------------------------------------
  // choices
  // ----------------------------------------------------------

  if (
    Array.isArray(obj.choices)
  ) {
    obj.choices =
      obj.choices.map(
        (choice) => {
          if (
            !choice ||
            typeof choice !== "object"
          ) {
            return choice;
          }

          // --------------------------------------------------
          // delta
          // --------------------------------------------------

          if (
            choice.delta &&
            typeof choice.delta === "object"
          ) {
            const delta =
              choice.delta;

            if (
              delta.content !== undefined &&
              delta.content !== null &&
              typeof delta.content !== "string"
            ) {
              delta.content =
                String(
                  delta.content
                );
            }

            if (
              delta.reasoning_content !== undefined &&
              delta.reasoning_content !== null &&
              typeof delta.reasoning_content !== "string"
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
            typeof choice.message === "object"
          ) {
            const message =
              choice.message;

            if (
              message.content !== undefined &&
              message.content !== null &&
              typeof message.content !== "string"
            ) {
              message.content =
                String(
                  message.content
                );
            }

            if (
              message.reasoning_content !== undefined &&
              message.reasoning_content !== null &&
              typeof message.reasoning_content !== "string"
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
// Build Cloudflare chat payload
// ============================================================

function buildChatPayload(
  reqBody,
  model,
  messages,
  isStream
) {
  const maxTokens =
    resolveMaxTokens(
      reqBody,
      model
    );

  const payload = {
    model,
    messages,
    stream: isStream,
    max_tokens: maxTokens
  };

  // ----------------------------------------------------------
  // Forward supported generation parameters
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
    const key of forwardParams
  ) {
    if (
      reqBody?.[key] !== undefined
    ) {
      payload[key] =
        reqBody[key];
    }
  }

  return payload;
}

// ============================================================
// CHAT COMPLETIONS
//
// Qwen3 + Granite
// ============================================================

app.post(
  "/v1/chat/completions",
  async (req, res) => {

    const isStream =
      req.body?.stream === true;

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

    const payload =
      buildChatPayload(
        req.body,
        model,
        messages,
        isStream
      );

    // --------------------------------------------------------
    // Debug log
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
      "RESOLVED max_tokens:",
      payload.max_tokens
    );

    console.log(
      "Message count:",
      messages.length
    );

    console.log(
      "Input chars:",
      messages.reduce(
        (total, m) =>
          total +
          m.content.length,
        0
      )
    );

    console.log(
      "stream_options:",
      req.body?.stream_options
    );

    console.log(
      "FINAL CLOUDFLARE PAYLOAD:",
      JSON.stringify({
        model: payload.model,
        stream: payload.stream,
        max_tokens:
          payload.max_tokens,
        temperature:
          payload.temperature,
        top_p:
          payload.top_p,
        top_k:
          payload.top_k,
        stream_options:
          payload.stream_options
      })
    );

    console.log(
      "=================================================="
    );

    // --------------------------------------------------------
    // Cloudflare URL
    // --------------------------------------------------------

    const cfUrl =
      `https://api.cloudflare.com/client/v4/accounts/` +
      `${CF_ACCOUNT}/ai/v1/chat/completions`;

    // --------------------------------------------------------
    // Abort upstream when client disconnects
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
            "CLIENT DISCONNECTED -> abort Cloudflare request"
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

        // Flush immediately
        if (
          typeof res.flushHeaders ===
          "function"
        ) {
          res.flushHeaders();
        }

        // ----------------------------------------------------
        // Cloudflare streaming request
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
        // Cloudflare error
        // ----------------------------------------------------

        if (!cfRes.ok) {

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
        // Empty body
        // ----------------------------------------------------

        if (!cfRes.body) {

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
        // SSE compatibility parser
        //
        // We parse SSE only so we can normalize Qwen
        // non-string delta.content.
        //
        // We preserve:
        //   usage
        //   finish_reason
        //   reasoning_content
        //   token_ids
        //   model
        //   id
        //   created
        //   other fields
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

          if (done) {
            break;
          }

          if (!value) {
            continue;
          }

          buffer +=
            decoder.decode(
              value,
              {
                stream: true
              }
            );

          // --------------------------------------------------
          // SSE normally uses blank line to terminate event.
          //
          // Process complete lines.
          // --------------------------------------------------

          const lines =
            buffer.split(
              /\r?\n/
            );

          // Keep incomplete final line
          buffer =
            lines.pop() || "";

          for (
            const line of lines
          ) {

            // ------------------------------------------------
            // Empty SSE separator
            // ------------------------------------------------

            if (
              line.trim() === ""
            ) {
              continue;
            }

            // ------------------------------------------------
            // SSE comment
            // ------------------------------------------------

            if (
              line.startsWith(":")
            ) {
              // We don't need to forward provider comments.
              continue;
            }

            // ------------------------------------------------
            // Ignore non-data SSE fields
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
              data === "[DONE]"
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

            } catch (e) {

              console.warn(
                "Skipping malformed SSE JSON:",
                data
              );

              continue;
            }

            // ------------------------------------------------
            // Normalize ONLY compatibility-sensitive fields
            // ------------------------------------------------

            chunk =
              normalizeChatResponseObject(
                chunk
              );

            // ------------------------------------------------
            // Forward normalized chunk
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
        // Flush remaining buffered data
        // ----------------------------------------------------

        buffer +=
          decoder.decode();

        if (
          buffer.trim()
        ) {

          const remaining =
            buffer.split(
              /\r?\n/
            );

          for (
            const line of remaining
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
              data === "[DONE]"
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

            } catch (e) {

              console.warn(
                "Skipping final malformed SSE JSON:",
                data
              );
            }
          }
        }

        // ----------------------------------------------------
        // Append DONE only if Cloudflare omitted it
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

      if (!cfRes.ok) {

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
      // Normalize JSON response
      //
      // This protects Cherry from the same content type issue
      // if Qwen ever returns numeric message.content.
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

      } catch (e) {

        // If Cloudflare returned something unexpected,
        // preserve the original response.
        res.setHeader(
          "Content-Type",
          "application/json"
        );

        return res.send(
          text
        );
      }

    } catch (e) {

      console.error(
        "CHAT ERROR:",
        e
      );

      // ------------------------------------------------------
      // AbortError = client disconnected
      // ------------------------------------------------------

      if (
        e?.name ===
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

      if (isStream) {

        if (
          !res.headersSent
        ) {

          return res
            .status(500)
            .json({
              error: {
                message:
                  e?.message ||
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
                      e?.message ||
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
            // Connection already closed
          }
        }

        return;
      }

      // ------------------------------------------------------
      // Non-streaming error
      // ------------------------------------------------------

      if (
        !res.headersSent
      ) {

        return res
          .status(500)
          .json({
            error: {
              message:
                e?.message ||
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
// Kept independent from Qwen / Granite chat route.
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

    // FLUX.2 Klein 4B
    form.append(
      "steps",
      "4"
    );

    // --------------------------------------------------------
    // IMG2IMG
    // --------------------------------------------------------

    if (imgInput) {

      const inputString =
        String(
          imgInput
        );

      const b64 =
        inputString.includes(",")
          ? inputString.split(",")[1]
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
    // Cloudflare FLUX endpoint
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

            // IMPORTANT:
            // Do not set Content-Type manually.
            // fetch() generates the multipart boundary.
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
    // Cloudflare FLUX error
    // --------------------------------------------------------

    if (!cfRes.ok) {

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
    // Parse image response
    // --------------------------------------------------------

    let data;

    try {

      data =
        JSON.parse(
          text
        );

    } catch (e) {

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

  } catch (e) {

    console.error(
      "FINAL IMAGE ERROR:",
      e
    );

    if (
      !res.headersSent
    ) {

      return res
        .status(500)
        .json({
          error: {
            message:
              e?.message ||
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
// Prevent default HTML:
// <pre>Bad Request</pre>
// ============================================================

app.use(
  (
    err,
    req,
    res,
    next
  ) => {

    if (
      err instanceof SyntaxError &&
      err?.status === 400 &&
      "body" in err
    ) {

      console.error(
        "INVALID JSON BODY:",
        err.message
      );

      return res
        .status(400)
        .json({
          error: {
            message:
              "Invalid JSON request body",

            detail:
              err.message
          }
        });
    }

    console.error(
      "UNHANDLED EXPRESS ERROR:",
      err
    );

    if (
      res.headersSent
    ) {
      return next(err);
    }

    return res
      .status(
        err?.status || 500
      )
      .json({
        error: {
          message:
            err?.message ||
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
      `V18 running on port ${PORT}`
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
      "Default Qwen max_tokens:",
      DEFAULT_MAX_TOKENS.QWEN
    );

    console.log(
      "Default Granite max_tokens:",
      DEFAULT_MAX_TOKENS.GRANITE
    );

    console.log(
      "=================================================="
    );
  }
);


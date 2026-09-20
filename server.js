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
// These are only used when the client DOES NOT provide
// max_tokens / max_completion_tokens.
//
// Client-supplied value always wins.
// ============================================================

const DEFAULT_MAX_TOKENS = {
  QWEN: 4096,
  GRANITE: 4096
};

// ============================================================
// Basic validation
// ============================================================

if (!CF_ACCOUNT) {
  console.warn("WARNING: CF_ACCOUNT_ID is not configured");
}

if (!CF_TOKEN) {
  console.warn("WARNING: CF_API_TOKEN is not configured");
}

// ============================================================
// Root / Health
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
    .send("ready V17 qwen+granite+flux");
});

// ============================================================
// OpenAI-compatible models endpoint
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

  // Compatibility aliases
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

  // Only allow Qwen / Granite in chat endpoint
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
//   1. max_completion_tokens
//   2. max_tokens
//   3. model-specific default
//
// IMPORTANT:
// If client supplies a valid value, DO NOT overwrite it.
// ============================================================

function resolveMaxTokens(reqBody, model) {
  let rawValue = null;

  // Newer OpenAI-compatible clients
  if (
    reqBody?.max_completion_tokens !== undefined &&
    reqBody?.max_completion_tokens !== null
  ) {
    rawValue = reqBody.max_completion_tokens;
  }

  // Traditional OpenAI parameter
  else if (
    reqBody?.max_tokens !== undefined &&
    reqBody?.max_tokens !== null
  ) {
    rawValue = reqBody.max_tokens;
  }

  // ----------------------------------------------------------
  // Client specified a value
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
  // No valid client value -> defaults
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
// Normalize messages
//
// Cherry Studio / OpenAI-compatible clients may send
// content either as a string or an array.
// ============================================================

function normalizeMessages(inputMessages) {
  if (!Array.isArray(inputMessages)) {
    return [];
  }

  return inputMessages
    .map((m) => {
      let content = m?.content;

      // --------------------------------------------------------
      // Array content
      // --------------------------------------------------------

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

      // --------------------------------------------------------
      // Normal string
      // --------------------------------------------------------

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
// Forward generation parameters
//
// These parameters are supported by Cloudflare's current
// Qwen3 / Granite model schemas.
//
// stream_options is forwarded because OpenAI-compatible clients
// such as Cherry Studio may send:
// {
//   "include_usage": true
// }
// ============================================================

function buildPayload(reqBody, model, messages, isStream) {
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
  // Supported generation parameters
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
    // Build Cloudflare payload
    // --------------------------------------------------------

    const payload =
      buildPayload(
        req.body,
        model,
        messages,
        isStream
      );

    // --------------------------------------------------------
    // IMPORTANT DEBUG LOG
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
          total + m.content.length,
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
    // Abort upstream if client disconnects
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
      // STREAM MODE
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

        // Flush HTTP headers immediately
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
        // Cloudflare error before stream starts
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
            // SSE-compatible error
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
        // Empty upstream body
        // ----------------------------------------------------

        if (!cfRes.body) {
          console.error(
            "CF STREAM ERROR: no response body"
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
        // RAW SSE RELAY
        //
        // IMPORTANT:
        //
        // We do NOT JSON.parse()
        // We do NOT modify delta.content
        // We do NOT modify usage
        // We do NOT modify reasoning_content
        // We do NOT rebuild SSE messages
        //
        // Cloudflare -> Render -> Cherry Studio
        // ----------------------------------------------------

        const reader =
          cfRes.body.getReader();

        let receivedDone =
          false;

        const decoder =
          new TextDecoder();

        let detectBuffer = "";

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

          // --------------------------------------------------
          // Relay original bytes
          // --------------------------------------------------

          if (
            !res.writableEnded
          ) {
            res.write(
              Buffer.from(value)
            );
          }

          // --------------------------------------------------
          // Detect [DONE]
          //
          // This is ONLY for logging.
          // The actual SSE data is never modified.
          // --------------------------------------------------

          try {
            const text =
              decoder.decode(
                value,
                {
                  stream: true
                }
              );

            detectBuffer +=
              text;

            if (
              detectBuffer.includes(
                "data: [DONE]"
              )
            ) {
              receivedDone =
                true;
            }

            // Prevent buffer growth
            if (
              detectBuffer.length >
              2000
            ) {
              detectBuffer =
                detectBuffer.slice(
                  -2000
                );
            }
          } catch (e) {
            // Detection failure must never break streaming.
          }
        }

        console.log(
          "CF STREAM ENDED. DONE RECEIVED:",
          receivedDone
        );

        // ----------------------------------------------------
        // IMPORTANT:
        //
        // Normally Cloudflare sends [DONE], which we already
        // relay unchanged.
        //
        // If Cloudflare closes without [DONE], append one
        // OpenAI-compatible DONE event.
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

        return;
      }

      // ======================================================
      // NON-STREAM MODE
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
              message: text
            }
          });
      }

      // ------------------------------------------------------
      // Successful response
      // ------------------------------------------------------

      res.setHeader(
        "Content-Type",
        "application/json"
      );

      res.send(text);

    } catch (e) {
      // ======================================================
      // Error handling
      // ======================================================

      console.error(
        "CHAT ERROR:",
        e
      );

      // Client disconnected
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
          res
            .status(500)
            .json({
              error: {
                message:
                  e?.message ||
                  "Streaming error"
              }
            });

        } else if (
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
            // Connection already closed.
          }
        }

        return;
      }

      // ------------------------------------------------------
      // Normal request error
      // ------------------------------------------------------

      if (
        !res.headersSent
      ) {
        res
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
// Kept separate from chat route.
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

    // FLUX.2 Klein 4B uses 4 inference steps
    form.append(
      "steps",
      "4"
    );

    // --------------------------------------------------------
    // IMG2IMG
    // --------------------------------------------------------

    if (imgInput) {
      const inputString =
        String(imgInput);

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
            type: "image/jpeg"
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
            // Do NOT set Content-Type manually.
            // fetch() creates the multipart boundary.
          },

          body: form
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
            message: text
          }
        });
    }

    // --------------------------------------------------------
    // Parse JSON
    // --------------------------------------------------------

    let data;

    try {
      data =
        JSON.parse(text);

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
    // OpenAI-style image response
    // --------------------------------------------------------

    res.json({
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
      res
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
// IMAGE ENDPOINTS
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
// JSON parser / application error handler
//
// This converts Express default HTML 400 responses into JSON.
// ============================================================

app.use(
  (err, req, res, next) => {
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
// Start
// ============================================================

app.listen(
  PORT,
  () => {
    console.log(
      "=================================================="
    );

    console.log(
      `V17 running on port ${PORT}`
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


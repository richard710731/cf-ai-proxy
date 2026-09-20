import express from "express";
import cors from "cors";

const app = express();

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

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
// Default max tokens
//
// Qwen3 official default = 2000
// Granite official default = 256
//
// We intentionally use 4096 for both so that long answers
// are not prematurely cut off.
//
// Frontend-supplied max_tokens always takes priority.
// ============================================================

const DEFAULT_MAX_TOKENS = {
  QWEN: 4096,
  GRANITE: 4096
};

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
    .send("ready V16 qwen+granite+flux");
});

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
// Helper: resolve model
// ============================================================

function resolveChatModel(inputModel) {
  let model = String(inputModel || MODELS.QWEN).trim();

  // Existing compatibility rules
  if (model.includes("llama") || model.includes("fast")) {
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
// Helper: get max_tokens
//
// Priority:
//   1. max_completion_tokens
//   2. max_tokens
//   3. model-specific default
//
// Cloudflare Workers AI expects max_tokens.
// ============================================================

function resolveMaxTokens(reqBody, model) {
  let value = null;

  if (reqBody?.max_completion_tokens != null) {
    value = Number(reqBody.max_completion_tokens);
  } else if (reqBody?.max_tokens != null) {
    value = Number(reqBody.max_tokens);
  }

  // If frontend supplied a valid value, use it.
  if (Number.isFinite(value) && value > 0) {
    // Integer only
    return Math.floor(value);
  }

  // Otherwise model-specific defaults
  if (model.includes("granite")) {
    return DEFAULT_MAX_TOKENS.GRANITE;
  }

  if (model.includes("qwen")) {
    return DEFAULT_MAX_TOKENS.QWEN;
  }

  return DEFAULT_MAX_TOKENS.QWEN;
}

// ============================================================
// Helper: normalize messages
// ============================================================

function normalizeMessages(inputMessages) {
  if (!Array.isArray(inputMessages)) {
    return [];
  }

  return inputMessages
    .map((m) => {
      let content = m?.content;

      // OpenAI-style content array
      if (Array.isArray(content)) {
        content = content
          .map((x) => {
            if (typeof x === "string") {
              return x;
            }

            if (x?.text != null) {
              return x.text;
            }

            if (x?.content != null) {
              return x.content;
            }

            return "";
          })
          .join("\n");
      }

      return {
        role: String(m?.role || "user"),
        content: String(content ?? "")
      };
    })
    .filter((m) => m.content.length > 0);
}

// ============================================================
// CHAT COMPLETIONS
// Qwen + Granite
// ============================================================

app.post("/v1/chat/completions", async (req, res) => {
  const isStream = req.body?.stream === true;

  const model = resolveChatModel(req.body?.model);
  const messages = normalizeMessages(req.body?.messages);
  const maxTokens = resolveMaxTokens(req.body, model);

  // ----------------------------------------------------------
  // Build Cloudflare payload
  // ----------------------------------------------------------

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
    "raw"
  ];

  for (const key of forwardParams) {
    if (req.body?.[key] !== undefined) {
      payload[key] = req.body[key];
    }
  }

  console.log(
    "CHAT REQUEST",
    JSON.stringify({
      model,
      stream: isStream,
      max_tokens: maxTokens,
      messageCount: messages.length,
      inputChars: messages.reduce(
        (sum, m) => sum + m.content.length,
        0
      )
    })
  );

  const cfUrl =
    `https://api.cloudflare.com/client/v4/accounts/` +
    `${CF_ACCOUNT}/ai/v1/chat/completions`;

  const controller = new AbortController();

  // ----------------------------------------------------------
  // If client disconnects, abort upstream Cloudflare request.
  // ----------------------------------------------------------

  res.on("close", () => {
    if (!res.writableEnded) {
      console.warn(
        "CLIENT DISCONNECTED -> abort Cloudflare request"
      );

      controller.abort();
    }
  });

  try {
    // ========================================================
    // STREAMING
    // ========================================================

    if (isStream) {
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
        "X-Accel-Buffering",
        "no"
      );

      res.setHeader(
        "Connection",
        "keep-alive"
      );

      // Flush headers immediately
      if (typeof res.flushHeaders === "function") {
        res.flushHeaders();
      }

      // ------------------------------------------------------
      // SSE heartbeat
      //
      // SSE comment lines are ignored by OpenAI-style clients.
      // Helps keep long-running connections alive.
      // ------------------------------------------------------

      const heartbeat = setInterval(() => {
        if (!res.writableEnded) {
          try {
            res.write(`: heartbeat ${Date.now()}\n\n`);
          } catch (e) {
            console.warn(
              "SSE heartbeat failed:",
              e.message
            );
          }
        }
      }, 15000);

      try {
        // ----------------------------------------------------
        // Call Cloudflare
        // ----------------------------------------------------

        const cfRes = await fetch(cfUrl, {
          method: "POST",

          headers: {
            Authorization: `Bearer ${CF_TOKEN}`,
            "Content-Type": "application/json",
            Accept: "text/event-stream"
          },

          body: JSON.stringify(payload),

          signal: controller.signal
        });

        console.log(
          "CF STREAM STATUS:",
          cfRes.status,
          cfRes.headers.get("content-type")
        );

        // ----------------------------------------------------
        // Cloudflare error
        // ----------------------------------------------------

        if (!cfRes.ok) {
          const text = await cfRes.text();

          console.error(
            "CF STREAM ERROR:",
            text
          );

          if (!res.writableEnded) {
            res.write(
              `data: ${JSON.stringify({
                error: {
                  message: text,
                  status: cfRes.status
                }
              })}\n\n`
            );

            res.end();
          }

          return;
        }

        // ----------------------------------------------------
        // Empty response
        // ----------------------------------------------------

        if (!cfRes.body) {
          console.error(
            "CF STREAM ERROR: empty response body"
          );

          if (!res.writableEnded) {
            res.write(
              `data: ${JSON.stringify({
                error: {
                  message:
                    "Cloudflare returned an empty response body"
                }
              })}\n\n`
            );

            res.end();
          }

          return;
        }

        // ----------------------------------------------------
        // IMPORTANT:
        //
        // Do NOT parse/rebuild the upstream SSE.
        //
        // Relay the original Cloudflare chunks directly.
        // ----------------------------------------------------

        const reader = cfRes.body.getReader();

        let sawDone = false;
        const decoder = new TextDecoder();
        let detectionBuffer = "";

        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          if (!value) {
            continue;
          }

          // --------------------------------------------------
          // RAW SSE RELAY
          // --------------------------------------------------

          if (!res.writableEnded) {
            res.write(value);
          }

          // --------------------------------------------------
          // Only inspect for [DONE].
          //
          // We do NOT modify the actual data.
          // --------------------------------------------------

          try {
            const chunkText = decoder.decode(
              value,
              { stream: true }
            );

            detectionBuffer += chunkText;

            if (
              detectionBuffer.includes("data: [DONE]")
            ) {
              sawDone = true;
            }

            // Prevent unbounded buffer growth
            if (detectionBuffer.length > 2000) {
              detectionBuffer =
                detectionBuffer.slice(-2000);
            }
          } catch (e) {
            // Detection is optional; never break streaming.
          }
        }

        // ----------------------------------------------------
        // If Cloudflare closed without [DONE],
        // provide OpenAI-compatible DONE.
        // ----------------------------------------------------

        if (
          !sawDone &&
          !res.writableEnded
        ) {
          res.write("data: [DONE]\n\n");
        }

        if (!res.writableEnded) {
          res.end();
        }

      } finally {
        clearInterval(heartbeat);
      }

      return;
    }

    // ========================================================
    // NON-STREAMING
    // ========================================================

    const cfRes = await fetch(cfUrl, {
      method: "POST",

      headers: {
        Authorization: `Bearer ${CF_TOKEN}`,
        "Content-Type": "application/json",
        Accept: "application/json"
      },

      body: JSON.stringify(payload),

      signal: controller.signal
    });

    console.log(
      "CF NON-STREAM STATUS:",
      cfRes.status,
      cfRes.headers.get("content-type")
    );

    const text = await cfRes.text();

    if (!cfRes.ok) {
      console.error(
        "CF ERROR:",
        text
      );

      return res.status(cfRes.status).json({
        error: {
          message: text
        }
      });
    }

    res.setHeader(
      "Content-Type",
      "application/json"
    );

    res.send(text);

  } catch (e) {
    console.error(
      "CHAT ERROR:",
      e
    );

    // --------------------------------------------------------
    // Client disconnected / AbortController
    // --------------------------------------------------------

    if (e?.name === "AbortError") {
      console.warn(
        "Cloudflare request aborted because client disconnected."
      );

      return;
    }

    // --------------------------------------------------------
    // Streaming error
    // --------------------------------------------------------

    if (isStream) {
      if (!res.headersSent) {
        res.status(500).json({
          error: {
            message:
              e?.message || "Streaming error"
          }
        });
      } else if (!res.writableEnded) {
        try {
          res.write(
            `data: ${JSON.stringify({
              error: {
                message:
                  e?.message ||
                  "Streaming error"
              }
            })}\n\n`
          );

          res.end();
        } catch (_) {
          // Connection already closed
        }
      }

      return;
    }

    // --------------------------------------------------------
    // Normal request error
    // --------------------------------------------------------

    if (!res.headersSent) {
      res.status(500).json({
        error: {
          message:
            e?.message ||
            "Request failed"
        }
      });
    }
  }
});

// ============================================================
// IMAGE GENERATION / EDITING
//
// FLUX.2 Klein 4B
//
// This section intentionally keeps your existing behavior
// separate from Qwen/Granite chat streaming.
// ============================================================

async function handleImage(req, res) {
  try {
    const model = MODELS.IMAGE;

    const prompt = String(
      req.body.prompt ||
      "full body photo of the same person, same face"
    );

    const size = String(
      req.body.size || "910x512"
    ).split("x");

    const width = String(
      parseInt(size[0], 10) || 910
    );

    const height = String(
      parseInt(size[1], 10) || 512
    );

    const imgInput =
      req.body.image ||
      req.body.image_b64;

    const form = new FormData();

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

    // FLUX.2 Klein is fixed at 4 steps
    form.append(
      "steps",
      "4"
    );

    // --------------------------------------------------------
    // img2img
    // --------------------------------------------------------

    if (imgInput) {
      const b64String = String(imgInput);

      const b64 = b64String.includes(",")
        ? b64String.split(",")[1]
        : b64String;

      const buffer = Buffer.from(
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
          req.body.strength || 0.5
        )
      );

      console.log(
        "KLEIN IMG2IMG MULTIPART, size",
        buffer.length
      );

    } else {

      console.log(
        "KLEIN TEXT2IMG MULTIPART"
      );
    }

    const cfUrl =
      `https://api.cloudflare.com/client/v4/accounts/` +
      `${CF_ACCOUNT}/ai/run/${model}`;

    const cfRes = await fetch(cfUrl, {
      method: "POST",

      headers: {
        Authorization:
          `Bearer ${CF_TOKEN}`
        // Do NOT manually set Content-Type.
        // fetch will generate the multipart boundary.
      },

      body: form
    });

    const text = await cfRes.text();

    console.log(
      "CF IMAGE STATUS",
      cfRes.status
    );

    if (!cfRes.ok) {
      console.error(
        "CF KLEIN ERROR:",
        text
      );

      return res
        .status(cfRes.status)
        .json({
          error: {
            message: text
          }
        });
    }

    let data;

    try {
      data = JSON.parse(text);
    } catch (e) {
      console.error(
        "FLUX JSON PARSE ERROR:",
        text
      );

      return res.status(502).json({
        error: {
          message:
            "Cloudflare returned invalid JSON"
        }
      });
    }

    res.json({
      created: Date.now(),

      data: [
        {
          b64_json:
            data.result?.image ||
            data.result
        }
      ]
    });

  } catch (e) {

    console.error(
      "FINAL IMAGE ERROR:",
      e
    );

    if (!res.headersSent) {
      res.status(500).json({
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
// Start server
// ============================================================

app.listen(
  PORT,
  () => {
    console.log(
      `V16 running on port ${PORT}`
    );

    console.log(
      "Models:",
      JSON.stringify(MODELS)
    );

    console.log(
      "Default max_tokens:",
      JSON.stringify(DEFAULT_MAX_TOKENS)
    );
  }
);


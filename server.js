import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ extended: true }));

const CF_ACCOUNT = process.env.CF_ACCOUNT_ID;
const CF_TOKEN = process.env.CF_API_TOKEN;
const PORT = process.env.PORT || 10000;

if (!CF_ACCOUNT ||!CF_TOKEN) {
  console.error("Missing CF_ACCOUNT_ID or CF_API_TOKEN");
}

app.get("/", (req, res) => {
  res.send(`Proxy OK - ${new Date().toISOString()} - Neurons check /v1/models`);
});

app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",
    data: [
      { id: "@cf/ibm-granite/granite-4.0-h-small", object: "model", owned_by: "ibm", name: "Granite 4.0 Small 8B" },
      { id: "@cf/ibm-granite/granite-4.0-h-micro", object: "model", owned_by: "ibm", name: "Granite 4.0 Micro 3B" },
      { id: "@cf/meta/llama-3.1-8b-instruct-fast", object: "model", owned_by: "meta", name: "Llama 3.1 8B Fast" },
      { id: "@cf/black-forest-labs/flux-2-klein-4b", object: "model", owned_by: "black-forest", name: "FLUX.2 Klein 4B" },
      { id: "@cf/leonardo/lucid-origin", object: "model", owned_by: "leonardo", name: "Lucid Origin" },
    ]
  });
});

// --- Chat Completions ---
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, stream } = req.body;
    if (!model) return res.status(400).json({ error: { message: "model required" } });

    const encodedModel = encodeURIComponent(model);
    const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${encodedModel}`;

    const cfRes = await fetch(cfUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${CF_TOKEN}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ messages, stream:!!stream })
    });

    // 統一錯誤格式，讓 Render 和 Worker 報一樣
    if (!cfRes.ok) {
      const errText = await cfRes.text();
      console.error("CF Error:", errText);
      let errJson;
      try { errJson = JSON.parse(errText); } catch {}
      const msg = errJson?.errors?.[0]?.message || errJson?.error || errText;
      return res.status(429).json({
        error: {
          message: `AiError: ${msg}`,
          type: "api_error",
          param: null,
          code: "cloudflare_api_error"
        }
      });
    }

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const reader = cfRes.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          try {
            const j = JSON.parse(payload);
            const content = j.response || j.result?.response || "";
            if (content) {
              const chunk = { id: "chatcmpl-" + Date.now(), object: "chat.completion.chunk", choices: [{ index: 0, delta: { content }, finish_reason: null }] };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }
          } catch {
            // 透傳
            res.write(line + "\n\n");
          }
        }
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      const data = await cfRes.json();
      const content = data.result?.response || data.result || "";
      res.json({
        id: "chatcmpl-" + Date.now(),
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }]
      });
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: { message: e.message, type: "server_error" } });
  }
});

// --- Images Generations (文生圖 + 圖生圖 + 細參數) ---
app.post("/v1/images/generations", async (req, res) => {
  try {
    let { model, prompt, size = "1024x512", steps, guidance, seed, image, strength, negative_prompt } = req.body;
    if (!model) model = "@cf/black-forest-labs/flux-2-klein-4b";

    const encodedModel = encodeURIComponent(model);
    const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${encodedModel}`;

    const [width, height] = size.split("x").map(Number);

    const payload = {
      prompt,
      width: width || 1024,
      height: height || 512,
      num_steps: steps || 20,
      guidance: guidance || 7.5,
    };
    if (seed) payload.seed = Number(seed);
    if (negative_prompt) payload.negative_prompt = negative_prompt;

    // 圖生圖
    if (image) {
      const b64 = image.includes(",")? image.split(",")[1] : image;
      payload.image = b64;
      payload.strength = strength? Number(strength) : 0.8;
      payload.num_steps = steps || 30;
    }

    console.log("Image:", model, size, "img2img:",!!image);

    const cfRes = await fetch(cfUrl, {
      method: "POST",
      headers: { Authorization: `Bearer ${CF_TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!cfRes.ok) {
      const errText = await cfRes.text();
      console.error("CF Image Error:", errText);
      let errJson;
      try { errJson = JSON.parse(errText); } catch {}
      const msg = errJson?.errors?.[0]?.message || errText;
      return res.status(429).json({
        error: { message: `AiError: ${msg}`, type: "api_error", code: "cloudflare_api_error" }
      });
    }

    const data = await cfRes.json();
    res.json({
      created: Math.floor(Date.now() / 1000),
      data: [{ b64_json: data.result.image, revised_prompt: prompt }],
      seed: data.result.seed
    });

  } catch (e) {
    console.error(e);
    res.status(500).json({ error: { message: e.message } });
  }
});

app.listen(PORT, () => console.log(`V6 Final running on ${PORT}`));

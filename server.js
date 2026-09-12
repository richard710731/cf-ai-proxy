import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json({ limit: "25mb" }));

const CF_ACCOUNT = process.env.CF_ACCOUNT_ID?.trim();
const CF_TOKEN = process.env.CF_API_TOKEN?.trim();
const PORT = process.env.PORT || 10000;

app.get("/", (req, res) => res.send("V7 OK " + new Date().toISOString()));
app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",
    data: [
      { id: "@cf/ibm-granite/granite-4.0-h-small", object: "model", owned_by: "ibm" },
      { id: "@cf/ibm-granite/granite-4.0-h-micro", object: "model", owned_by: "ibm" },
      { id: "@cf/meta/llama-3.1-8b-instruct-fast", object: "model", owned_by: "meta" },
      { id: "@cf/black-forest-labs/flux-2-klein-4b", object: "model", owned_by: "black-forest" },
    ]
  });
});

app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, stream } = req.body;
    // 改用新版 v1 端點，model 放 body，不放 URL，就不會 No route
    const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/v1/chat/completions`;

    const cfRes = await fetch(cfUrl, {
      method: "POST",
      headers: { Authorization: `Bearer ${CF_TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages, stream: !!stream })
    });

    if (!cfRes.ok) {
      const errText = await cfRes.text();
      console.error("CF CHAT ERROR:", errText);
      return res.status(cfRes.status).json({ error: { message: `AiError: ${errText}`, type: "api_error", code: "cloudflare_api_error" } });
    }

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      // 新版 v1 回來的已經是 OpenAI 格式，直接 pipe 就好
      cfRes.body.pipe(res);
    } else {
      const data = await cfRes.text();
      res.setHeader("Content-Type", "application/json");
      res.send(data);
    }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: { message: e.message } });
  }
});

app.post("/v1/images/generations", async (req, res) => {
  try {
    const { model, prompt, size = "1024x512", steps, guidance, seed, image, strength } = req.body;
    const [w, h] = size.split("x").map(Number);
    
    // 生圖還是用舊的 /ai/run/ 但不要 encode
    const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${model}`;

    const payload = { prompt, width: w, height: h, num_steps: steps || 20, guidance: guidance || 7.5 };
    if (seed) payload.seed = Number(seed);
    if (image) {
      payload.image = image.includes(",") ? image.split(",")[1] : image;
      payload.strength = Number(strength) || 0.8;
    }

    const cfRes = await fetch(cfUrl, {
      method: "POST",
      headers: { Authorization: `Bearer ${CF_TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!cfRes.ok) {
      const errText = await cfRes.text();
      console.error("CF IMAGE ERROR:", errText);
      return res.status(429).json({ error: { message: `AiError: ${errText}`, type: "api_error", code: "cloudflare_api_error" } });
    }
    const data = await cfRes.json();
    res.json({ created: Date.now(), data: [{ b64_json: data.result.image }] });
  } catch (e) {
    res.status(500).json({ error: { message: e.message } });
  }
});

app.listen(PORT, () => console.log(`V7 running on ${PORT}`));
  }
});

app.listen(PORT, () => console.log(`V6 Final running on ${PORT}`));

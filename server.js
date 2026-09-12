import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json({ limit: "25mb" }));

const CF_ACCOUNT = process.env.CF_ACCOUNT_ID? process.env.CF_ACCOUNT_ID.trim() : "";
const CF_TOKEN = process.env.CF_API_TOKEN? process.env.CF_API_TOKEN.trim() : "";
const PORT = process.env.PORT || 10000;

app.get("/", function(req, res) {
  res.send("V8 OK " + new Date().toISOString());
});

app.get("/v1/models", function(req, res) {
  res.json({
    object: "list",
    data: [
      { id: "@cf/meta/llama-3.1-8b-instruct-fast", object: "model", owned_by: "meta" },
      { id: "@cf/ibm-granite/granite-4.0-h-micro", object: "model", owned_by: "ibm" },
      { id: "@cf/black-forest-labs/flux-2-klein-4b", object: "model", owned_by: "black-forest" }
    ]
  });
});

app.post("/v1/chat/completions", async function(req, res) {
  try {
    const model = req.body.model;
    const messages = req.body.messages;
    const stream = req.body.stream;
    const cfUrl = "https://api.cloudflare.com/client/v4/accounts/" + CF_ACCOUNT + "/ai/v1/chat/completions";

    const cfRes = await fetch(cfUrl, {
      method: "POST",
      headers: {
        Authorization: "Bearer " + CF_TOKEN,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ model: model, messages: messages, stream:!!stream })
    });

    if (!cfRes.ok) {
      const errText = await cfRes.text();
      console.error("CF CHAT ERROR:", errText);
      return res.status(cfRes.status).json({
        error: { message: "AiError: " + errText, type: "api_error", code: "cloudflare_api_error" }
      });
    }

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      const reader = cfRes.body.getReader();
      while (true) {
        const result = await reader.read();
        if (result.done) break;
        res.write(result.value);
      }
      res.end();
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

app.post("/v1/images/generations", async function(req, res) {
  try {
    const model = req.body.model || "@cf/black-forest-labs/flux-2-klein-4b";
    const prompt = req.body.prompt;
    const size = req.body.size || "1024x512";
    const parts = size.split("x");
    const w = parseInt(parts[0], 10);
    const h = parseInt(parts[1], 10);

    const cfUrl = "https://api.cloudflare.com/client/v4/accounts/" + CF_ACCOUNT + "/ai/run/" + model;

    const payload = {
      prompt: prompt,
      width: w || 1024,
      height: h || 512,
      num_steps: req.body.steps || 20,
      guidance: req.body.guidance || 7.5
    };

    if (req.body.seed) payload.seed = Number(req.body.seed);
    if (req.body.image) {
      const img = req.body.image;
      payload.image = img.includes(",")? img.split(",")[1] : img;
      payload.strength = Number(req.body.strength) || 0.8;
    }

    const cfRes = await fetch(cfUrl, {
      method: "POST",
      headers: { Authorization: "Bearer " + CF_TOKEN, "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!cfRes.ok) {
      const errText = await cfRes.text();
      console.error("CF IMAGE ERROR:", errText);
      return res.status(429).json({ error: { message: "AiError: " + errText } });
    }

    const data = await cfRes.json();
    res.json({ created: Date.now(), data: [{ b64_json: data.result.image }] });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: { message: e.message } });
  }
});

app.listen(PORT, function() {
  console.log("V8 running on " + PORT);
});

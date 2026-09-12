import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json({ limit: "25mb" }));

const CF_ACCOUNT = (process.env.CF_ACCOUNT_ID || "").trim();
const CF_TOKEN = (process.env.CF_API_TOKEN || "").trim();
const PORT = process.env.PORT || 10000;

function normalizeMessages(msgs) {
  return (msgs || []).map(m => {
    let content = m.content;
    if (Array.isArray(content)) {
      content = content.map(c => {
        if (typeof c === "string") return c;
        return c.text || c.content || "";
      }).join("\n");
    }
    if (typeof content !== "string") content = String(content || "");
    return { role: m.role || "user", content: content };
  }).filter(m => m.content);
}

app.get("/", (req,res)=>res.send("V9 OK "+new Date().toISOString()));
app.get("/v1/models", (req,res)=>{
  res.json({ object:"list", data:[
      { id: "@cf/meta/llama-3.1-8b-instruct-fast", object: "model", owned_by: "meta" },
      { id: "@cf/ibm-granite/granite-4.0-h-micro", object: "model", owned_by: "ibm" },
      {id:"@cf/black-forest-labs/flux-1-schnell", object:"model", owned_by:"black-forest"},
      { id: "@cf/black-forest-labs/flux-2-klein-4b", object: "model", owned_by: "black-forest" }
  ]});
});

app.post("/v1/chat/completions", async (req,res)=>{
  try{
    let { model, messages, stream } = req.body;
    messages = normalizeMessages(messages);
    
    // granite 用新版 v1, flux 那些不用
    const isChatModel = model.includes("granite") || model.includes("llama") || model.includes("gemma");
    const cfUrl = isChatModel
      ? `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/v1/chat/completions`
      : `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${model}`;

    const body = isChatModel ? { model, messages, stream: !!stream } : { messages };

    const cfRes = await fetch(cfUrl, {
      method:"POST",
      headers:{ Authorization:`Bearer ${CF_TOKEN}`, "Content-Type":"application/json" },
      body: JSON.stringify(body)
    });

    if(!cfRes.ok){
      const t = await cfRes.text();
      console.error("CF ERROR:", t);
      return res.status(cfRes.status).json({ error:{ message:"AiError: "+t, type:"api_error", code:"cloudflare_api_error" }});
    }

    if(stream && isChatModel){
      res.setHeader("Content-Type","text/event-stream");
      res.setHeader("Cache-Control","no-cache");
      const reader = cfRes.body.getReader();
      while(true){
        const {done,value} = await reader.read();
        if(done) break;
        res.write(value);
      }
      res.end();
    } else if(stream) {
      // 舊版 /ai/run/ 的串流處理
      res.setHeader("Content-Type","text/event-stream");
      const reader = cfRes.body.getReader();
      const decoder = new TextDecoder();
      while(true){
        const {done,value} = await reader.read();
        if(done) break;
        const chunk = decoder.decode(value,{stream:true});
        for(const line of chunk.split("\n")){
          if(line.startsWith("data:")){
            try{
              const j = JSON.parse(line.slice(5));
              if(j.response) res.write(`data: ${JSON.stringify({choices:[{delta:{content:j.response}}]})}\n\n`);
            }catch{ res.write(line+"\n\n"); }
          }
        }
      }
      res.write("data: [DONE]\n\n"); res.end();
    } else {
      const text = await cfRes.text();
      res.setHeader("Content-Type","application/json");
      res.send(text);
    }
  }catch(e){
    console.error(e);
    res.status(500).json({error:{message:e.message}});
  }
});

async function handleImage(req, res) {
  try {
    const model = "@cf/black-forest-labs/flux-2-klein-4b";
    const prompt = String(req.body.prompt || "full body photo of the same person, same face");
    const size = (req.body.size || "910x512").split("x");
    const width = String(parseInt(size[0]) || 910);
    const height = String(parseInt(size[1]) || 512);
    const imgInput = req.body.image || req.body.image_b64;

    const form = new FormData();
    form.append("prompt", prompt);
    form.append("width", width);
    form.append("height", height);
    form.append("steps", "8");

    if (imgInput) {
      const b64 = String(imgInput).includes(",")? String(imgInput).split(",")[1] : String(imgInput);
      const buffer = Buffer.from(b64, "base64");
      // 關鍵：檔名要是 input.jpg，type 要 image/jpeg
      form.append("image", new Blob([buffer], {type:"image/jpeg"}), "input.jpg");
      form.append("strength", String(req.body.strength || 0.5));
      console.log("KLEIN IMG2IMG MULTIPART, size", buffer.length);
    } else {
      console.log("KLEIN TEXT2IMG MULTIPART");
    }

    const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${model}`;
    const cfRes = await fetch(cfUrl, {
      method: "POST",
      headers: { Authorization: `Bearer ${CF_TOKEN}` }, // multipart 不要自己加 Content-Type
      body: form
    });

    const text = await cfRes.text();
    console.log("CF STATUS", cfRes.status);
    if (!cfRes.ok) { console.error("CF KLEIN ERROR:", text); return res.status(cfRes.status).json({ error:{message:text} }); }

    const data = JSON.parse(text);
    res.json({ created: Date.now(), data:[{ b64_json: data.result?.image || data.result }] });

  } catch (e) { console.error("FINAL ERROR", e); res.status(500).json({error:{message:e.message}}); }
}

app.post("/v1/images/generations", handleImage);
app.post("/v1/images/edits", handleImage);

app.listen(PORT,()=>console.log("V9 running "+PORT));

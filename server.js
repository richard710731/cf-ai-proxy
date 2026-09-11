import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json({limit: "25mb"}));
app.use(express.urlencoded({extended:true}));

const CF_ACCOUNT = process.env.CF_ACCOUNT_ID;
const CF_TOKEN = process.env.CF_API_TOKEN;

app.get("/v1/models", (req,res)=>{
  res.json({
    object:"list",
    data:[
      {id:"@cf/ibm/granite-3.1-8b-instruct", object:"model", owned_by:"ibm"},
      {id:"@cf/ibm/granite-3.0-8b-instruct", object:"model", owned_by:"ibm"},
      {id:"@cf/black-forest-labs/flux-2-klein-4b", object:"model", owned_by:"black-forest"},
      {id:"@cf/leonardo/lucid-origin", object:"model", owned_by:"leonardo"},
    ]
  });
});

// --- Chat ---
app.post("/v1/chat/completions", async (req,res)=>{
  const { model, messages, stream } = req.body;
  const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${model}`;
  const cfRes = await fetch(cfUrl, {
    method:"POST",
    headers: { Authorization: `Bearer ${CF_TOKEN}`, "Content-Type":"application/json" },
    body: JSON.stringify({ messages, stream:!!stream })
  });
  if(!cfRes.ok) return res.status(500).json({error: await cfRes.text()});

  if(stream){
    res.setHeader("Content-Type","text/event-stream");
    res.setHeader("Cache-Control","no-cache");
    const reader = cfRes.body.getReader();
    const decoder = new TextDecoder();
    while(true){
      const { done, value } = await reader.read();
      if(done) break;
      const chunk = decoder.decode(value, {stream:true});
      for(const line of chunk.split("\n")){
        if(!line.startsWith("data:")) continue;
        try{
          const j = JSON.parse(line.slice(5));
          const content = j.response || "";
          if(content) res.write(`data: ${JSON.stringify({choices:[{delta:{content}}]})}\n\n`);
        }catch{ res.write(line+"\n\n"); }
      }
    }
    res.write("data: [DONE]\n\n"); res.end();
  } else {
    const data = await cfRes.json();
    res.json({ choices:[{ message:{ role:"assistant", content:data.result.response }}] });
  }
});

// --- Image Gen + Img2Img ---
app.post("/v1/images/generations", async (req,res)=>{
  try{
    let { model, prompt, size="1024x512", n=1, guidance, steps, seed, image, strength, negative_prompt } = req.body;

    // Cherry Studio 傳圖生圖時，會把 image 放在 image 欄位，可能是 base64 或 URL
    const [width, height] = size.split('x').map(Number);

    const cfPayload = {
      prompt,
      width, height,
      // 把細參數全部透傳給 Cloudflare，沒填就用預設
      num_steps: steps || 20,
      guidance: guidance || 7.5,
      seed: seed || undefined,
    };
    if(negative_prompt) cfPayload.negative_prompt = negative_prompt;

    // 圖生圖
    if(image){
      // 如果是 data:image/...;base64,xxx 只取後面
      if(image.includes(",")) image = image.split(",")[1];
      cfPayload.image = image; // base64
      cfPayload.strength = strength || 0.8; // 0.1~1，越低越像原圖
      cfPayload.num_steps = steps || 30; // 圖生圖建議 30步
    }

    console.log("Image req:", model, size, "img2img:",!!image);

    const cfRes = await fetch(`https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${model}`,{
      method:"POST",
      headers:{ Authorization:`Bearer ${CF_TOKEN}`, "Content-Type":"application/json"},
      body: JSON.stringify(cfPayload)
    });
    const data = await cfRes.json();
    if(!cfRes.ok) return res.status(500).json(data);

    res.json({
      created: Date.now(),
      data: [{ b64_json: data.result.image }],
      // 把 seed 回傳，方便你下次復現
      seed: data.result.seed
    });
  }catch(e){
    res.status(500).json({error: e.message});
  }
});

app.listen(10000, ()=>console.log("V4 running"));

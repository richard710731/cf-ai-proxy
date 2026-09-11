import express from "express";
import cors from "cors";
const app = express();
app.use(cors());
app.use(express.json({limit: "25mb"}));

const CF_ACCOUNT = process.env.CF_ACCOUNT_ID;
const CF_TOKEN = process.env.CF_API_TOKEN;

app.get("/v1/models", (req,res)=>{
  res.json({
    object:"list",
    data:[
      {id:"@cf/ibm-granite/granite-4.0-h-micro", object:"model", owned_by:"ibm"},
      {id:"@cf/meta/llama-3.1-8b-instruct-fast", object:"model", owned_by:"cloudflare"},
      {id:"@cf/black-forest-labs/flux-2-klein-4b", object:"model", owned_by:"black-forest"}
    ]
  });
});

app.post("/v1/chat/completions", async (req,res)=>{
  const { model, messages, stream } = req.body;
  const encodedModel = encodeURIComponent(model); // 修復7000錯誤
  const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${encodedModel}`;
  const cfRes = await fetch(cfUrl, {
    method:"POST",
    headers: { Authorization: `Bearer ${CF_TOKEN}`, "Content-Type":"application/json" },
    body: JSON.stringify({ messages, stream:!!stream })
  });
  if(!cfRes.ok){
    const t = await cfRes.text();
    console.log("CF ERROR:", t);
    return res.status(500).json({error: t});
  }
  if(stream){
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
    const data = await cfRes.json();
    res.json({ choices:[{ message:{ content:data.result.response }}] });
  }
});

app.post("/v1/images/generations", async (req,res)=>{
  const { model, prompt, size="1024x512", steps, guidance, seed, image, strength } = req.body;
  const [w,h] = size.split('x').map(Number);
  const encodedModel = encodeURIComponent(model); // 修復7000錯誤
  const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${encodedModel}`;

  const payload = { prompt, width:w, height:h, num_steps: steps||20, guidance: guidance||7.5 };
  if(seed) payload.seed = seed;
  if(image){
    payload.image = image.includes(",")? image.split(",")[1] : image;
    payload.strength = strength || 0.8;
  }
  const cfRes = await fetch(cfUrl,{
    method:"POST",
    headers:{ Authorization:`Bearer ${CF_TOKEN}`, "Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  const data = await cfRes.json();
  if(!cfRes.ok) return res.status(500).json(data);
  res.json({ created: Date.now(), data:[{ b64_json: data.result.image }] });
});

app.listen(10000, ()=>console.log("V5 fixed"));

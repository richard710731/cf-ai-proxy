import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json({limit: "10mb"}));

const CF_ACCOUNT = process.env.CF_ACCOUNT_ID;
const CF_TOKEN = process.env.CF_API_TOKEN;

app.get("/v1/models", async (req,res)=>{
  const r = await fetch(`https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/models/search`, {
    headers: { Authorization: `Bearer ${CF_TOKEN}` }
  });
  const data = await r.json();
  res.json({ object:"list", data: data.result.map(m=>({id:m.name, object:"model", owned_by:"cloudflare"})) });
});

app.post("/v1/chat/completions", async (req,res)=>{
  const { model, messages, stream } = req.body;
  const cfRes = await fetch(`https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${model}`, {
    method:"POST",
    headers: { Authorization: `Bearer ${CF_TOKEN}`, "Content-Type":"application/json" },
    body: JSON.stringify({ messages, stream })
  });

  if (stream) {
    res.setHeader("Content-Type","text/event-stream");
    cfRes.body.pipeTo(new WritableStream({
      write(chunk){ res.write(chunk); }
    }));
  } else {
    const data = await cfRes.json();
    res.json({
      id:"chatcmpl-"+Date.now(),
      object:"chat.completion",
      choices:[{ message:{ role:"assistant", content:data.result.response } }]
    });
  }
});

app.listen(10000, ()=>console.log("running on 10000"));

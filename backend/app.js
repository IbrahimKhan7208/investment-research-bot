// Must be first, before any other import. @langchain/langgraph calls
// crypto.randomUUID() as a bare global (not require("crypto")), which
// Node only exposes globally by default on v20+. Render's default Node
// version is older than that, causing "crypto is not defined" on every
// graph.stream()/graph.invoke() call in production while working fine
// locally on your Node v24. This polyfill makes it work regardless of
// which Node version actually ends up running — the engines pin in
// package.json (see package-json-engines-snippet.txt) is the real fix,
// this is the belt-and-suspenders backup so it can't silently break again.
import { webcrypto } from 'node:crypto';
if (!globalThis.crypto) {
  globalThis.crypto = webcrypto;
}

import express from 'express';
import cors from 'cors';
import { graph } from './agent.js';

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Investment Research API is running' });
});

app.post('/api/research', async (req, res) => {
  try {
    const { question } = req.body;
    const result = await graph.invoke({ originalQuestion: question });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/research/stream", async (req, res) => {
  const question = req.query.question;
  if (!question) return res.status(400).end();

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const send = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  try {
    // "custom" carries every live emit() call from agent.js as it happens
    // — this is what makes sub-agent lanes and the router step feel
    // real-time instead of arriving in one lump when the whole node
    // finishes. "updates" still carries final state snapshots (evidence,
    // finalOutput, verdicts, companyBriefs) once each node completes.
    const stream = await graph.stream({ originalQuestion: question }, { streamMode: ["updates", "custom"] });

    for await (const [mode, data] of stream) {
      if (mode === "custom") {
        send("trace", { entry: data });
        continue;
      }

      const nodeName = Object.keys(data)[0];
      const payload = data[nodeName];

      if (nodeName === "subAgent" && payload?.companyBriefs?.length) {
        send("step", { node: "subAgent", company: payload.companyBriefs[0].company, payload });
      } else {
        send("step", { node: nodeName, payload });
      }
    }

    send("done", {});
  } catch (err) {
    send("error", { message: err.message });
  } finally {
    res.end();
  }
});

app.listen(PORT, () => {
  console.log(`\nInvestment Research API`);
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
});
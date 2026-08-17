import express from 'express';
import cors from 'cors';
import { graph } from './agent.js';

const app = express()
const PORT = 3000

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

// GET, not POST — EventSource (SSE) only supports GET requests
app.get("/api/research/stream", async (req, res) => {
  const question = req.query.question;
  if (!question) return res.status(400).end();

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    // streamMode: "updates" yields { nodeName: whateverThatNodeReturned }
    // after each node finishes — exactly the granularity needed to show
    // "classifier done -> rag running -> rag done -> synthesizer running" etc.
    const stream = await graph.stream({ originalQuestion: question }, { streamMode: "updates" });

    for await (const update of stream) {
      const nodeName = Object.keys(update)[0];
      send("step", { node: nodeName, payload: update[nodeName] });
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
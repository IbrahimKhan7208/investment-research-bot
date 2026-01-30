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

app.listen(PORT, () => {
  console.log(`\nInvestment Research API`);
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
});
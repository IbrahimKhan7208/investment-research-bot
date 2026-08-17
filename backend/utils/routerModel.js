import "dotenv/config";
import { Client } from "@gradio/client";

const SPACE_ID = "IbrahimKhan7208/investment-research-router-space";
let client = null;

async function getClient() {
  if (!client) {
    client = await Client.connect(SPACE_ID, {
      token: process.env.HF_TOKEN,
    });
  }
  return client;
}

export async function routeInvestmentQuestion({
  question,
  temperature = 0.1,
  maxNewTokens = 1024,
  retries = 3,
}) {
  let lastError;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const c = await getClient();

      const result = await c.predict("/route", {
        question,
        temperature,
        max_new_tokens: maxNewTokens,
      });

      const routing = result.data[0];

      if (routing?.error) {
        // Deterministic (greedy) model - retrying the same question won't
        // produce a different output. Fail fast instead of wasting retries.
        throw Object.assign(new Error(`Router model error: ${routing.error}`), { retryable: false });
      }

      if (!routing?.subQuestions || !routing?.requiredTools) {
        throw Object.assign(new Error("Router returned unexpected shape: " + JSON.stringify(routing)), { retryable: false });
      }

      return routing;
    } catch (err) {
      lastError = err;
      console.warn(`[routeInvestmentQuestion] attempt ${attempt} failed: ${err.message}`);

      if (err.retryable === false) break;   // model-output failure, don't retry

      // reset the cached client - a network/connection failure may mean
      // the cached connection itself is stale, not just this one call
      client = null;

      if (attempt < retries) {
        const backoffMs = attempt * 5000;
        await new Promise((r) => setTimeout(r, backoffMs));
      }
    }
  }

  throw new Error(`routeInvestmentQuestion failed: ${lastError.message}`);
}
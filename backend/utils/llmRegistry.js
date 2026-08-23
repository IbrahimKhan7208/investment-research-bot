import dotenv from "dotenv";
dotenv.config();
import { ChatOpenAI } from "@langchain/openai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";

// ─────────────────────────────────────────────────────────────────────────
// Mirrors agent.js exactly — same models, same config. If these fail here,
// they fail in the graph too. Nothing added, nothing changed.
// ─────────────────────────────────────────────────────────────────────────

export const draftingLLM = new ChatOpenAI({
  model: "openai/gpt-oss-20b:free",
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});

export const verifierLLM = new ChatGoogleGenerativeAI({
  model: "gemini-3.5-flash",
  apiKey: process.env.GOOGLE_API_KEY,
});

export const extractionLLM = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0.1,
  maxRetries: 2,
});

export const registry = {
  drafting: draftingLLM,
  verifier: verifierLLM,
  extraction: extractionLLM,
};

// ─────────────────────────────────────────────────────────────────────────
// safeInvoke — wraps .invoke to catch the exact bug from your trace:
// generations[0] coming back as an empty array (200 OK, empty choices).
// Use this in agent.js in place of raw llm.invoke() once we know which
// models need it.
// ─────────────────────────────────────────────────────────────────────────

export async function safeInvoke(llm, messages, label = "llm") {
  try {
    const res = await llm.invoke(messages);
    if (!res || res.content === undefined) {
      throw new Error(`[${label}] Response had no content — likely empty choices array upstream.`);
    }
    return res;
  } catch (err) {
    if (err.message?.includes("Cannot read properties of undefined")) {
      throw new Error(
        `[${label}] Provider returned 200 with empty choices (rate limit or invalid message shape). Original: ${err.message}`
      );
    }
    throw err;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Test harness — reproduces agent.js's actual call shape (system-only)
// AND a system+user shape, per model. This isolates whether the crash is:
//   (a) message-shape rejection (system-only not accepted)
//   (b) free-tier rate limit / quota exhaustion
//   (c) something else entirely
// ─────────────────────────────────────────────────────────────────────────

async function runCase(name, llm, messages, caseLabel) {
  const tag = `[${name}/${caseLabel}]`;
  try {
    const res = await llm.invoke(messages);
    const generations = res?.content;
    if (generations === undefined) {
      console.log(`${tag} FAIL — got a response object but .content is undefined (empty choices upstream)`);
      console.log(`${tag} raw response:`, JSON.stringify(res)?.slice(0, 300));
      return false;
    }
    console.log(`${tag} PASS —`, JSON.stringify(generations).slice(0, 150));
    return true;
  } catch (err) {
    console.log(`${tag} FAIL —`, err.message);
    if (err.response?.status) console.log(`${tag} HTTP status:`, err.response.status);
    if (err.response?.data) console.log(`${tag} body:`, JSON.stringify(err.response.data)?.slice(0, 400));
    if (err.status) console.log(`${tag} status field:`, err.status);
    if (err.error) console.log(`${tag} error field:`, JSON.stringify(err.error).slice(0, 400));
    return false;
  }
}

export async function testAll() {
  const results = {};

  for (const [name, llm] of Object.entries(registry)) {
    results[name] = {};

    // Case 1: system-only — exact shape used everywhere in agent.js
    results[name].systemOnly = await runCase(name, llm, [
      { role: "system", content: "Reply with exactly one word: OK" },
    ], "system-only");

    // Case 2: system + user — the shape most providers actually expect
    results[name].systemPlusUser = await runCase(name, llm, [
      { role: "system", content: "You are a terse assistant." },
      { role: "user", content: "Reply with exactly one word: OK" },
    ], "system+user");

    console.log(""); // spacer between models
  }

  console.log("Summary:", JSON.stringify(results, null, 2));

  // Diagnosis
  for (const [name, r] of Object.entries(results)) {
    if (!r.systemOnly && r.systemPlusUser) {
      console.log(`>>> [${name}] fails on system-only messages but works with system+user. This is a MESSAGE SHAPE bug, not rate limiting. Fix: add a user turn in every agent.js call to this model.`);
    } else if (!r.systemOnly && !r.systemPlusUser) {
      console.log(`>>> [${name}] fails on both shapes. Likely rate limit / quota / auth issue, not message shape.`);
    }
  }

  return results;
}

import { pathToFileURL } from "node:url";

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  testAll();
}
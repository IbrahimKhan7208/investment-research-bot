import dotenv from "dotenv";
dotenv.config();
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { ChatGroq } from "@langchain/groq";
import { ChatOpenAI } from "@langchain/openai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Annotation, StateGraph, Send } from "@langchain/langgraph";
import { CohereEmbeddings, CohereRerank } from "@langchain/cohere";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { marked } from "marked";
import TerminalRenderer from "marked-terminal";
import { TavilySearch } from "@langchain/tavily";
import { StockTool } from "./utils/stockTool.js";
import { routeInvestmentQuestion } from "./utils/routerModel.js";
import { resolveCik } from "./utils/secEdgar.js";
import { verifyGrounding } from "./utils/verifier.js";
import { withRetry } from "./utils/withRetry.js";
import { MAX_VERIFIER_ITERATIONS, budgetExceeded, MAX_FINALIZE_VERIFIER_ITERATIONS } from "./utils/loopControl.js";
import { mechanicalCoherenceCheck } from "./utils/mechanicalChecks.js";
import { planGapRetrieval } from "./utils/gapEscalation.js";
import { buildEvidenceBlock, CITATION_RULES, SCALE_PRESERVATION_RULE } from "./utils/evidenceFormat.js";
import { guardedVerify, resetVerifierBudget } from "./utils/verifierGuard.js";
import chalk from "chalk";

marked.setOptions({ renderer: new TerminalRenderer() });

// ─────────────────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────────────────

const AgentState = Annotation.Root({
  originalQuestion: Annotation(),
  subQuestions: Annotation(),
  requiredTools: Annotation(),

  evidence: Annotation({
    reducer: (existing = [], update = []) => existing.concat(update),
    default: () => [],
  }),
  toolsExecuted: Annotation({
    reducer: (existing = [], update = []) => [...new Set(existing.concat(update))],
    default: () => [],
  }),
  agentTrace: Annotation({
    reducer: (existing = [], update = []) => existing.concat(update),
    default: () => [],
  }),
  companyBriefs: Annotation({
    reducer: (existing = [], update = []) => existing.concat(update),
    default: () => [],
  }),

  finalOutput: Annotation(),
  verifierVerdict: Annotation(),
  verifierAttempts: Annotation({ default: () => 0 }),
  gapFillRounds: Annotation({ default: () => 0 }),

  subAgentMaxAttempts: Annotation({
    reducer: (existing = 0, update = 0) => Math.max(existing, update),
    default: () => 0,
  }),

  runStartTime: Annotation(),
});

// ─────────────────────────────────────────────────────────────────────────
// CROSS-COMPANY SCOPE GUARDS — the actual fix for this turn.
//
// Root cause: in a multi-company question, the router produces ONE
// sub-question whose TEXT still names every company ("What was NVIDIA and
// AMD's data center revenue growth..."), even though `companies` is
// correctly split per sub-agent at dispatch. The retrieval FILTER was
// already scoped correctly (each sub-agent only ever queries its own
// company's vectors) — that part was never broken. What was broken: the
// LOCAL VERIFIER was being shown the raw, unscoped originalQuestion when
// checking a company's brief, so it would reasonably conclude "the
// sibling company's data is missing" and generate a gap asking for it —
// and that gap then got executed literally, as a real RAG/WEB call,
// inside a scope where it could never possibly succeed. Every "provided
// excerpts do not contain [other company]" answer in the evidence panel
// is one of these — a real, paid retrieval + extraction call, guaranteed
// to fail before it even ran.
//
// Two-layer fix:
//  1. scopeQuestionToCompany — tell the verifier explicitly it's grading
//     ONE company's brief only, so it stops generating the gap in the
//     first place.
//  2. gapMentionsOtherCompany — a hard backstop. Even a well-instructed
//     model doesn't follow instructions 100% of the time; this drops any
//     gap that slips through BEFORE it reaches a tool call, not after.
// ─────────────────────────────────────────────────────────────────────────

function scopeQuestionToCompany(questionText, company, allCompanies) {
  const others = allCompanies.filter(c => c !== company);
  if (others.length === 0) return questionText;
  return `${questionText}\n\n(Note: this brief covers ${company} only. ${others.join(", ")} ${others.length > 1 ? "are" : "is"} covered in a separate, independent brief — do not treat missing ${others.join("/")} data as a gap in THIS brief, and do not request evidence about ${others.join("/")}. If ${others.join("/")} is mentioned in the retrieved evidence, that's incidental — judge only what this brief claims about ${company}.)`;
}

function gapMentionsOtherCompany(gap, otherCompanies) {
  if (!otherCompanies?.length) return false;
  const text = `${gap.missing || ""} ${gap.suggestedQuery || ""}`.toLowerCase();
  return otherCompanies.some(c => text.includes(String(c).toLowerCase()));
}

// ─────────────────────────────────────────────────────────────────────────
// STREAMING HELPERS
// ─────────────────────────────────────────────────────────────────────────

function emit(config, entry) {
  config?.writer?.(entry);
}

function scopedConfig(config, company) {
  if (!config?.writer) return config;
  return {
    ...config,
    writer: (entry) => {
      if (entry?.node && !entry.node.startsWith(`subAgent:${company}:`)) {
        config.writer({ ...entry, node: `subAgent:${company}:${entry.node}` });
      } else {
        config.writer(entry);
      }
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────
// MODEL HARNESS
// ─────────────────────────────────────────────────────────────────────────

const extractionLLM = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0.1,
  maxRetries: 2,
});

const draftingLLM = new ChatOpenAI({
  model: "openai/gpt-oss-120b",
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});

const verifierLLM = new ChatGoogleGenerativeAI({
  model: "gemini-3.5-flash",
  apiKey: process.env.GOOGLE_API_KEY,
});

// ─────────────────────────────────────────────────────────────────────────
// SHARED CLIENTS
// ─────────────────────────────────────────────────────────────────────────

const MAX_XBRL_FACTS_PER_COMPANY_YEAR = 15;
const CANDIDATES_PER_COMPANY_YEAR = 12;
const FINAL_NARRATIVE_CHUNKS_AFTER_RERANK = 6;
const SOURCE_TYPE_PRIORITY = { "xbrl-fact": 0, "table-prose": 1, narrative: 2 };

const embeddings = new CohereEmbeddings({
  model: "embed-english-v3.0",
  apiKey: process.env.COHERE_API_KEY,
});

const pineconeClient = new PineconeClient();
const pineconeIndex = pineconeClient.Index(process.env.PINECONE_INDEX);

let vectorStorePromise = null;
function getVectorStore() {
  if (!vectorStorePromise) {
    vectorStorePromise = PineconeStore.fromExistingIndex(embeddings, { pineconeIndex });
  }
  return vectorStorePromise;
}

const reranker = new CohereRerank({
  apiKey: process.env.COHERE_API_KEY,
  model: "rerank-english-v3.0",
  topN: FINAL_NARRATIVE_CHUNKS_AFTER_RERANK,
});

const webTool = new TavilySearch({
  maxResults: 5,
  topic: "news",
  searchDepth: "basic",
  includeAnswer: false,
});

const stockTool = new StockTool();

// ─────────────────────────────────────────────────────────────────────────
// CLASSIFIER
// ─────────────────────────────────────────────────────────────────────────

async function classifierNode(state, config) {
  resetVerifierBudget();
  emit(config, { node: "router", phase: "thinking" });

  const { subQuestions, requiredTools } = await routeInvestmentQuestion({
    question: state.originalQuestion,
  });

  console.log("\nTo answer this question, the following tools will be used: ", requiredTools);
  emit(config, { node: "router", phase: "done", requiredTools, subQuestions });

  return {
    subQuestions,
    requiredTools,
    runStartTime: Date.now(),
    verifierAttempts: 0,
    gapFillRounds: 0,
  };
}

// ─────────────────────────────────────────────────────────────────────────
// TOOL NODES
// ─────────────────────────────────────────────────────────────────────────

async function ragNode(state, config) {
  console.log("\n[RAG] Searching documents...\n");
  const vectorStore = await getVectorStore();
  const newTrace = [];
  const newEvidence = [];

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "RAG") continue;

    emit(config, { node: "rag", phase: "start", question: subQ.question });

    try {
      const companies = [];
      for (const companyName of subQ.companies) {
        const resolved = await resolveCik(companyName);
        if (!resolved) {
          console.warn(`   [RAG] Could not resolve "${companyName}" to a ticker — filtering on raw name, likely 0 results.`);
        }
        companies.push(resolved ? resolved.ticker : companyName);
      }

      const currentYear = new Date().getFullYear();
      const years = subQ.years.length > 0 ? subQ.years : [currentYear, currentYear - 1];
      const searchQuery = subQ.searchQuery || subQ.question;

      const xbrlFactChunks = [];
      for (const company of companies) {
        for (const year of years) {
          const result = await withRetry(
            () => vectorStore.similaritySearch(searchQuery, MAX_XBRL_FACTS_PER_COMPANY_YEAR, {
              company, year, sourceType: "xbrl-fact",
            }),
            { label: `pinecone xbrl-fact ${company} ${year}` }
          );
          xbrlFactChunks.push(...result);
        }
      }

      const narrativeCandidates = [];
      for (const company of companies) {
        for (const year of years) {
          let result = await withRetry(
            () => vectorStore.similaritySearch(searchQuery, CANDIDATES_PER_COMPANY_YEAR, { company, year }),
            { label: `pinecone narrative ${company} ${year}` }
          );
          let narrative = result.filter((doc) => doc.metadata.sourceType !== "xbrl-fact");

          if (narrative.length === 0) {
            result = await withRetry(
              () => vectorStore.similaritySearch(searchQuery, CANDIDATES_PER_COMPANY_YEAR, { company }),
              { label: `pinecone narrative fallback ${company}` }
            );
            narrative = result.filter((doc) => doc.metadata.sourceType !== "xbrl-fact");
          }
          narrativeCandidates.push(...narrative);
        }
      }

      console.log(chalk.green(`   ${xbrlFactChunks.length} xbrl-fact chunks (guaranteed), ${narrativeCandidates.length} narrative/table-prose candidates, reranking...\n`));

      const rerankedNarrative = narrativeCandidates.length > 0
        ? await withRetry(() => reranker.compressDocuments(narrativeCandidates, subQ.question), { label: "cohere rerank" })
        : [];

      const allChunks = [...xbrlFactChunks, ...rerankedNarrative].sort(
        (a, b) => SOURCE_TYPE_PRIORITY[a.metadata.sourceType] - SOURCE_TYPE_PRIORITY[b.metadata.sourceType]
      );

      const traceEntry = {
        node: "rag",
        question: subQ.question,
        xbrlFactCount: xbrlFactChunks.length,
        narrativeCandidateCount: narrativeCandidates.length,
        rerankedCount: rerankedNarrative.length,
      };
      newTrace.push(traceEntry);
      emit(config, traceEntry);

      console.log(chalk.green(`   ${allChunks.length} total chunks\n`));

      const SYSTEM_PROMPT = `You are a financial research assistant.

Answer the question using ONLY the provided document excerpts.
Do NOT use outside knowledge.
Do NOT speculate.

Question: ${subQ.question}

Documents:
${allChunks.map((doc, i) => `
[${i + 1}] ${doc.metadata.company} ${doc.metadata.year} — ${doc.metadata.section} (${doc.metadata.sourceType}):
${doc.pageContent}
`).join("\n")}

Instructions:
- Provide the factual answer.
- Include specific figures when available.
- If both an "xbrl-fact" and a "table-prose"/"narrative" excerpt give the same figure, prefer the xbrl-fact value — it comes from SEC's structured data, not parsed HTML.
- Cite sources using this format ONLY: [Company Year, Section] — e.g. [NVDA 2026, Item 7].
- Do NOT use any other bracket, footnote, or numeric citation marker (no 【N】-style tags, no bare [1][2] numbering). If you are not certain which exact section a fact comes from, omit the citation for that fact rather than guessing.
- Do NOT add opinions or analysis beyond the documents.

Answer:`;

      const answerResponse = await extractionLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);

      newEvidence.push({
        question: subQ.question,
        tool: "RAG",
        answer: answerResponse.content,
        sources: allChunks.map((doc) => ({
          company: doc.metadata.company,
          year: doc.metadata.year,
          section: doc.metadata.section,
          sourceType: doc.metadata.sourceType,
          formType: doc.metadata.formType,
          filingDate: doc.metadata.filingDate,
        })),
      });
    } catch (error) {
      console.log(chalk.red(`   [RAG] Failed for "${subQ.question}": ${error.message}`));
      const errEntry = { node: "rag", question: subQ.question, error: error.message };
      newEvidence.push({
        question: subQ.question,
        tool: "RAG",
        answer: `Unable to retrieve data for this sub-question (${error.message}). Treat this topic as unanswered rather than answered.`,
        sources: [],
      });
      newTrace.push(errEntry);
      emit(config, errEntry);
    }
  }

  return {
    evidence: newEvidence,
    toolsExecuted: ["RAG"],
    agentTrace: newTrace,
  };
}

async function webNode(state, config) {
  console.log("\n[WEB] Searching web...");

  const newTrace = [];
  const newEvidence = [];

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "WEB") continue;

    emit(config, { node: "web", phase: "start", question: subQ.question });

    try {
      const query = subQ.companies?.length
        ? `${subQ.question} (${subQ.companies.join(", ")})`
        : subQ.question;

      const response = await withRetry(() => webTool.invoke({ query }), { label: "tavily search" });

      const searchResults = response.results.map((result) => ({
        title: result.title,
        url: result.url,
        content: result.content,
      }));

      const traceEntry = { node: "web", question: subQ.question, query, articlesFound: searchResults.length };
      newTrace.push(traceEntry);
      emit(config, traceEntry);

      console.log(chalk.green(`   Retrieved ${searchResults.length} articles\n`));

      const SYSTEM_PROMPT = `You are a financial research assistant.

Answer this question using ONLY the provided search results.

Question: ${subQ.question}

Search Results:
${searchResults.map((result, i) => `[${i + 1}] ${result.title}\n${result.content}`).join("\n\n")}

Provide a clear, factual answer. Do NOT use outside knowledge.
Cite sources inline using [1], [2], etc. matching the numbered search results above.

Answer:`;

      const answer = await extractionLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);

      newEvidence.push({
        question: subQ.question,
        tool: "WEB",
        answer: answer.content,
        sources: searchResults.map((result) => ({ title: result.title, url: result.url })),
      });
    } catch (error) {
      console.log(chalk.red(`   [WEB] Failed for "${subQ.question}": ${error.message}`));
      const errEntry = { node: "web", question: subQ.question, error: error.message };
      newEvidence.push({
        question: subQ.question,
        tool: "WEB",
        answer: `Unable to retrieve data for this sub-question (${error.message}). Treat this topic as unanswered rather than answered.`,
        sources: [],
      });
      newTrace.push(errEntry);
      emit(config, errEntry);
    }
  }

  return {
    evidence: newEvidence,
    toolsExecuted: ["WEB"],
    agentTrace: newTrace,
  };
}

async function stockNode(state, config) {
  console.log("\n[STOCK] Fetching market data...");

  const newTrace = [];
  const newEvidence = [];

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "STOCK") continue;

    emit(config, { node: "stock", phase: "start", question: subQ.question });

    try {
      const stockTrace = {};
      const stockData = await stockTool.answerQuestion(subQ.question, subQ.companies, stockTrace);
      const traceEntry = { node: "stock", question: subQ.question, ...stockTrace, resultCount: Array.isArray(stockData) ? stockData.length : 0 };
      newTrace.push(traceEntry);
      emit(config, traceEntry);

      if (!stockData || (Array.isArray(stockData) && stockData.length === 0)) {
        throw new Error("No data returned from API");
      }

      const formattedData = Array.isArray(stockData)
        ? stockData.map((d, idx) => {
            if (d.price !== undefined) {
              return `[${idx + 1}] ${d.ticker}
                Price: $${d.price.toFixed(2)} ${d.currency}
                Change: ${d.change >= 0 ? "+" : ""}$${d.change.toFixed(2)} (${d.changePercent.toFixed(2)}%)
                Volume: ${d.volume.toLocaleString()}
                Market Cap: $${(d.marketCap / 1e9).toFixed(2)}B`;
            } else if (d.percentChange !== undefined) {
              return `[${idx + 1}] ${d.ticker} - ${d.period} Performance
                Start: $${d.startPrice.toFixed(2)} → End: $${d.endPrice.toFixed(2)}
                Change: ${d.priceChange >= 0 ? "+" : ""}$${d.priceChange.toFixed(2)} (${d.percentChange}%)`;
            }
          }).join("\n\n")
        : String(stockData);

      const SYSTEM_PROMPT = `You are a financial data analyst.

Answer this question using ONLY the provided stock market data.

Question: ${subQ.question}

Stock Data:
${formattedData}

Provide a clear, factual answer with specific numbers. Compare data if multiple companies mentioned.

Answer:`;

      const answer = await extractionLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);

      newEvidence.push({
        question: subQ.question,
        tool: "STOCK",
        answer: answer.content,
        sources: Array.isArray(stockData)
          ? stockData.map((d) => ({ ticker: d.ticker, price: d.price || d.endPrice, source: "Yahoo Finance" }))
          : [],
      });
    } catch (error) {
      console.log(`Error: ${error.message}\n`);
      const errEntry = { node: "stock", question: subQ.question, error: error.message };
      newEvidence.push({
        question: subQ.question,
        tool: "STOCK",
        answer: "Unable to retrieve stock data. The ticker might be invalid or the service is temporarily unavailable.",
        sources: [],
      });
      newTrace.push(errEntry);
      emit(config, errEntry);
    }
  }

  return {
    evidence: newEvidence,
    toolsExecuted: ["STOCK"],
    agentTrace: newTrace,
  };
}

// ─────────────────────────────────────────────────────────────────────────
// SOURCE APPENDIX
// ─────────────────────────────────────────────────────────────────────────

function renderSourceAppendix(evidence) {
  const webSources = (evidence || [])
    .filter(e => e.tool === "WEB")
    .flatMap(e => e.sources || []);
  if (webSources.length === 0) return "";
  return "\n\n**Sources**\n" + webSources
    .map((s, i) => `- [W${i + 1}] [${s.title}](${s.url})`)
    .join("\n");
}

// ─────────────────────────────────────────────────────────────────────────
// SINGLE-ENTITY SYNTHESIS + VERIFICATION PATH
// ─────────────────────────────────────────────────────────────────────────

async function synthesizerNode(state) {
  console.log("\n[SYNTHESIZER] Combining evidence...\n");

  const SYSTEM_PROMPT = `You are a financial research analyst. Synthesize a comprehensive answer based on the evidence.

Original Question: "${state.originalQuestion}"

Evidence Collected:
${buildEvidenceBlock(state.evidence)}

Instructions:
- Directly answer the user's question.
- Combine information across sub-questions logically.
- Cite sources using the format that matches each source's tool, exactly as listed above:
${CITATION_RULES}
${SCALE_PRESERVATION_RULE}
- Highlight key comparisons or insights.
- Clearly state limitations or missing data if applicable.
- Do NOT introduce new facts not present in the evidence.
- At the end, list the tools used to answer this question as a short reference line.

Final Answer:`;

  const finalResponse = await draftingLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);

  return { finalOutput: finalResponse.content + renderSourceAppendix(state.evidence) };
}

async function verifierNode(state, config) {
  console.log("\n[VERIFIER] Checking grounding...\n");
  emit(config, { node: "verifier", phase: "start" });

  const verdict = await guardedVerify({
    llm: verifierLLM,
    mode: "single",
    question: state.originalQuestion,
    evidenceBundle: state.evidence,
    draftText: state.finalOutput,
  }, "verifier");

  console.log(chalk.yellow(`   Verdict: ${verdict.verdict} — ${verdict.reasoning}\n`));

  return {
    verifierVerdict: verdict,
    agentTrace: [{
      node: "verifier",
      verdict: verdict.verdict,
      reasoning: verdict.reasoning,
      evidenceGapsCount: verdict.evidenceGaps?.length ?? 0,
      unsupportedCount: verdict.claimsChecked?.filter(c => !c.supported).length ?? 0,
      conflictsCount: verdict.conflicts?.length ?? 0,
    }],
  };
}

async function gapFillNode(state, config) {
  console.log("\n[GAP-FILL] Re-invoking tools for evidence gaps...\n");
  emit(config, { node: "gapFill", phase: "start" });

  const roundNumber = (state.gapFillRounds || 0) + 1;
  const planned = planGapRetrieval(state.verifierVerdict?.evidenceGaps || [], roundNumber);
  const abandoned = planned.filter(g => g.giveUp);
  const toRun = planned.filter(g => !g.giveUp);

  const gapSubQuestions = toRun.map((g) => ({
    question: g.missing,
    tool: g.tool,
    companies: g.companies || [],
    years: g.years || [],
    searchQuery: g.suggestedQuery,
  }));

  let working = { subQuestions: gapSubQuestions, evidence: [], agentTrace: [], toolsExecuted: [] };
  const neededTools = [...new Set(gapSubQuestions.map(q => q.tool))];

  for (const tool of neededTools) {
    const result = tool === "RAG" ? await ragNode(working, config)
                  : tool === "WEB" ? await webNode(working, config)
                  : await stockNode(working, config);
    working = {
      ...working,
      evidence: [...working.evidence, ...result.evidence],
      agentTrace: [...working.agentTrace, ...result.agentTrace],
      toolsExecuted: [...new Set([...working.toolsExecuted, ...result.toolsExecuted])],
    };
  }

  const escalationTrace = toRun.filter(g => g.escalated).map(g =>
    ({ node: "gapFill:escalated", from: g.originalTool, to: "WEB", gap: g.missing }));
  const abandonedTrace = abandoned.map(g =>
    ({ node: "gapFill:abandoned", gap: g.missing, reason: `already tried ${g.suggestedTool}, no further tool to escalate to` }));

  return {
    evidence: working.evidence,
    agentTrace: [...working.agentTrace, ...escalationTrace, ...abandonedTrace],
    toolsExecuted: working.toolsExecuted,
    verifierAttempts: (state.verifierAttempts || 0) + 1,
    gapFillRounds: roundNumber,
  };
}

async function resynthesizeNode(state, config) {
  console.log(chalk.magenta("\n[RESYNTHESIZE] Correcting previous answer...\n"));
  emit(config, { node: "resynthesize", phase: "start" });

  const verdict = state.verifierVerdict;
  let correctionInstruction = "";

  if (verdict.verdict === "unsupported_claim") {
    const bad = (verdict.claimsChecked || []).filter(c => !c.supported).map(c => `- "${c.claim}"`).join("\n");
    correctionInstruction = `The following claims were NOT supported by evidence and must be removed or rewritten to only state what evidence supports — do not replace them with new unsupported inferences:\n${bad}`;
  } else if (verdict.verdict === "conflicting_evidence") {
    const conf = (verdict.conflicts || []).map(c => `- "${c.claimA}" (${c.sourceA}) vs "${c.claimB}" (${c.sourceB})`).join("\n");
    correctionInstruction = `The evidence contains a conflict your previous answer did not surface. Explicitly state it rather than silently picking one side:\n${conf}`;
  }

  const SYSTEM_PROMPT = `You are a financial research analyst. You previously produced this answer:
"""
${state.finalOutput}
"""

${correctionInstruction}

Regenerate the answer using ONLY this evidence:
${buildEvidenceBlock(state.evidence)}

Keep the same citation format rules as before:
${CITATION_RULES}
${SCALE_PRESERVATION_RULE}

Final Answer:`;

  const finalResponse = await draftingLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);

  return {
    finalOutput: finalResponse.content + renderSourceAppendix(state.evidence),
    verifierAttempts: (state.verifierAttempts || 0) + 1,
    agentTrace: [{ node: "resynthesize", correctedVerdict: verdict.verdict }],
  };
}

async function capReachedNode(state, config) {
  console.log(chalk.red("\n[CAP-REACHED] Retry limit hit — returning best-effort answer with caveat.\n"));
  emit(config, { node: "capReached", phase: "start" });

  const verdict = state.verifierVerdict || {};
  let detail;
  if (verdict.verdict === "needs_more_evidence") {
    detail = `Unresolved evidence gaps: ${(verdict.evidenceGaps || []).map(g => g.missing).join("; ") || "unspecified"}`;
  } else if (verdict.verdict === "unsupported_claim") {
    detail = `Unresolved unsupported claims: ${(verdict.claimsChecked || []).filter(c => !c.supported).map(c => c.claim).join("; ") || "unspecified"}`;
  } else if (verdict.verdict === "conflicting_evidence") {
    detail = `Unresolved evidence conflicts: ${(verdict.conflicts || []).map(c => `${c.claimA} vs ${c.claimB}`).join("; ") || "unspecified"}`;
  } else {
    detail = "verification limit reached";
  }

  const caveat = `\n\n---\n*This answer was generated after reaching the verification retry limit. ${detail}. Treat unconfirmed figures with extra caution.*`;

  return {
    finalOutput: (state.finalOutput || "") + caveat,
    verifierVerdict: { ...verdict, verdict: "cap_reached" },
    agentTrace: [{ node: "capReached", attempts: state.verifierAttempts, originalVerdict: verdict.verdict }],
  };
}

// ─────────────────────────────────────────────────────────────────────────
// MULTI-ENTITY PATH
// ─────────────────────────────────────────────────────────────────────────

async function synthesizeCompanyBrief(originalQuestion, company, evidence) {
  const SYSTEM_PROMPT = `You are a financial research analyst producing a brief on ${company} only, as part of a larger multi-company answer to: "${originalQuestion}"

Evidence Collected (for ${company} only):
${buildEvidenceBlock(evidence)}

Instructions:
- Answer only what the evidence supports for ${company}.
- Cite sources using the format matching each source's tool:
${CITATION_RULES}
${SCALE_PRESERVATION_RULE}
- Do NOT introduce facts not present in the evidence.
- Do NOT compare ${company} to other companies — that happens later, in a separate merge step.

Brief:`;

  const response = await draftingLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);
  return response.content;
}

async function resynthesizeCompanyBrief(originalQuestion, company, evidence, previousBrief, verdict) {
  let correctionInstruction = "";
  if (verdict.verdict === "unsupported_claim") {
    const bad = (verdict.claimsChecked || []).filter(c => !c.supported).map(c => `- "${c.claim}"`).join("\n");
    correctionInstruction = `Remove or correct these unsupported claims:\n${bad}`;
  } else if (verdict.verdict === "conflicting_evidence") {
    const conf = (verdict.conflicts || []).map(c => `- "${c.claimA}" (${c.sourceA}) vs "${c.claimB}" (${c.sourceB})`).join("\n");
    correctionInstruction = `Explicitly surface this conflict instead of silently resolving it:\n${conf}`;
  }

  const SYSTEM_PROMPT = `You previously wrote this brief on ${company}:
"""
${previousBrief}
"""

${correctionInstruction}

Regenerate the brief using ONLY this evidence:
${buildEvidenceBlock(evidence)}

Citation rules unchanged:
${CITATION_RULES}
${SCALE_PRESERVATION_RULE}

Brief:`;

  const response = await draftingLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);
  return response.content;
}

// One sub-agent, one company. `scopedQuestion` and `otherCompanies` are new
// — see the header comment on scopeQuestionToCompany/gapMentionsOtherCompany
// for why. `originalQuestion` is kept for the brief-writing prompt (its own
// instructions already restrict scope: "producing a brief on {company}
// only"), but `scopedQuestion` — NOT `originalQuestion` — is what the local
// verifier is graded against, since that's the one place the cross-company
// gap was actually being generated.
async function subAgentNode(payload, config) {
  const { company, subQuestions, originalQuestion, scopedQuestion, otherCompanies, runStartTime } = payload;
  console.log(chalk.cyan(`\n[SUB-AGENT: ${company}] Starting...\n`));

  const cfg = scopedConfig(config, company);
  emit(cfg, { node: "starting" });

  let local = { subQuestions, evidence: [], agentTrace: [], toolsExecuted: [] };
  const neededTools = [...new Set(subQuestions.map(q => q.tool))];
  for (const tool of neededTools) {
    const result = tool === "RAG" ? await ragNode(local, cfg)
                  : tool === "WEB" ? await webNode(local, cfg)
                  : await stockNode(local, cfg);
    local = {
      subQuestions,
      evidence: [...local.evidence, ...result.evidence],
      agentTrace: [...local.agentTrace, ...result.agentTrace],
      toolsExecuted: [...new Set([...local.toolsExecuted, ...result.toolsExecuted])],
    };
  }

  let briefText = await synthesizeCompanyBrief(originalQuestion, company, local.evidence);
  let attempts = 0;
  let gapRoundsLocal = 0;

  while (true) {
    emit(cfg, { node: "verifying" });

    const verdict = await guardedVerify({
      llm: verifierLLM,
      mode: "local",
      question: scopedQuestion || originalQuestion, // ← the actual fix: verifier now grades against a question that explicitly excludes the sibling company
      company,
      evidenceBundle: local.evidence,
      draftText: briefText,
    }, `subAgent:${company}`);

    const verdictEntry = {
      node: `subAgent:${company}:verifier`,
      verdict: verdict.verdict,
      reasoning: verdict.reasoning,
    };
    local.agentTrace.push(verdictEntry);
    emit(config, verdictEntry);

    console.log(chalk.yellow(`   [${company}] verdict: ${verdict.verdict}`));

    if (verdict.verdict === "sufficient" || verdict.verdict === "verification_unavailable") {
      if (verdict.verdict === "verification_unavailable") {
        briefText += `\n\n*[${company} brief: verifier unavailable — ${verdict.reasoning}]*`;
        const unavailEntry = { node: `subAgent:${company}:verifierUnavailable` };
        local.agentTrace.push(unavailEntry);
        emit(config, unavailEntry);
      }
      break;
    }

    attempts++;
    if (attempts >= MAX_VERIFIER_ITERATIONS || budgetExceeded({ agentTrace: local.agentTrace, runStartTime })) {
      briefText += `\n\n*[${company} brief: verification retry limit reached — remaining issue: ${verdict.verdict}. Treat with caution.]*`;
      const capEntry = { node: `subAgent:${company}:capReached`, attempts };
      local.agentTrace.push(capEntry);
      emit(config, capEntry);
      break;
    }

    if (verdict.verdict === "needs_more_evidence") {
      gapRoundsLocal++;

      // Backstop: drop any gap that asks about a sibling company BEFORE it
      // ever reaches a tool call. This is what stops the exact waste seen
      // in production — a gap literally titled "NVIDIA FY2024 data center
      // revenue growth" being executed inside AMD's own scoped retrieval,
      // guaranteed to return "not found" every time.
      const rawGaps = verdict.evidenceGaps || [];
      const inScopeGaps = rawGaps.filter(g => !gapMentionsOtherCompany(g, otherCompanies));
      const outOfScopeGaps = rawGaps.filter(g => gapMentionsOtherCompany(g, otherCompanies));

      outOfScopeGaps.forEach(g => {
        const e = { node: `subAgent:${company}:abandoned`, gap: g.missing, reason: "out of scope — belongs to another company's brief, not retried" };
        local.agentTrace.push(e);
        emit(config, e);
      });

      if (inScopeGaps.length === 0) {
        // The entire "gap" was a cross-company artifact — nothing real is
        // actually missing from THIS brief. Stop here instead of spending
        // another round chasing nothing.
        break;
      }

      const planned = planGapRetrieval(inScopeGaps, gapRoundsLocal);
      const abandoned = planned.filter(g => g.giveUp);
      const toRun = planned.filter(g => !g.giveUp);

      const gapSubQs = toRun.map(g => ({
        question: g.missing,
        tool: g.tool,
        companies: [company],
        years: g.years || [],
        searchQuery: g.suggestedQuery,
      }));

      let gapWorking = { subQuestions: gapSubQs, evidence: [], agentTrace: [], toolsExecuted: [] };
      const gapTools = [...new Set(gapSubQs.map(q => q.tool))];
      for (const t of gapTools) {
        const result = t === "RAG" ? await ragNode(gapWorking, cfg)
                      : t === "WEB" ? await webNode(gapWorking, cfg)
                      : await stockNode(gapWorking, cfg);
        gapWorking = {
          subQuestions: gapSubQs,
          evidence: [...gapWorking.evidence, ...result.evidence],
          agentTrace: [...gapWorking.agentTrace, ...result.agentTrace],
          toolsExecuted: [...new Set([...gapWorking.toolsExecuted, ...result.toolsExecuted])],
        };
      }

      local.evidence = [...local.evidence, ...gapWorking.evidence];
      local.agentTrace.push(...gapWorking.agentTrace);

      toRun.filter(g => g.escalated).forEach(g => {
        const e = { node: `subAgent:${company}:escalated`, from: g.originalTool, to: "WEB", gap: g.missing };
        local.agentTrace.push(e);
        emit(config, e);
      });
      abandoned.forEach(g => {
        const e = { node: `subAgent:${company}:abandoned`, gap: g.missing, reason: `already tried ${g.suggestedTool}` };
        local.agentTrace.push(e);
        emit(config, e);
      });

      briefText = await synthesizeCompanyBrief(originalQuestion, company, local.evidence);
    } else {
      const resynthEntry = { node: `subAgent:${company}:resynthesize`, correctedVerdict: verdict.verdict };
      local.agentTrace.push(resynthEntry);
      emit(config, resynthEntry);
      briefText = await resynthesizeCompanyBrief(originalQuestion, company, local.evidence, briefText, verdict);
    }
  }

  const yearsCited = [...new Set(
    local.evidence.flatMap(e => (e.sources || []).map(s => s.year).filter(Boolean))
  )];

  return {
    evidence: local.evidence,
    agentTrace: local.agentTrace,
    toolsExecuted: local.toolsExecuted,
    companyBriefs: [{ company, yearsCited, text: briefText }],
    subAgentMaxAttempts: attempts,
  };
}

async function globalCheckNode(state, config) {
  const briefs = state.companyBriefs || [];

  if (briefs.length <= 1) {
    return { agentTrace: [{ node: "globalCheck", skipped: true, reason: "single brief, nothing to reconcile" }] };
  }

  console.log(chalk.cyan("\n[GLOBAL-CHECK] Checking cross-company coherence...\n"));
  emit(config, { node: "globalCheck", phase: "start" });

  const mechanicalFlags = mechanicalCoherenceCheck(briefs);

  const verdict = await guardedVerify({
    llm: verifierLLM,
    mode: "global",
    originalQuestion: state.originalQuestion,
    briefs,
    mechanicalFlags,
  }, "globalCheck");

  console.log(chalk.yellow(`   Global verdict: ${verdict.verdict} — ${verdict.reasoning}\n`));

  return {
    verifierVerdict: verdict,
    agentTrace: [{
      node: "globalCheck",
      verdict: verdict.verdict,
      reasoning: verdict.reasoning,
      mechanicalFlagsCount: mechanicalFlags.length,
    }],
  };
}

async function finalizeMultiEntityNode(state, config) {
  console.log("\n[FINALIZE] Merging company briefs into final answer...\n");
  emit(config, { node: "finalizeMultiEntity", phase: "start" });

  const briefs = state.companyBriefs || [];
  const verdict = state.verifierVerdict;

  let correctionNote = "";
  if (verdict?.verdict === "conflicting_evidence" && verdict.conflicts?.length) {
    correctionNote = `\n\nNote: explicitly surface these cross-company conflicts in your answer rather than silently resolving them:\n${verdict.conflicts.map(c => `- "${c.claimA}" (${c.sourceA}) vs "${c.claimB}" (${c.sourceB})`).join("\n")}`;
  }

  const SYSTEM_PROMPT = `You are a financial research analyst. Combine these already-verified per-company briefs into one coherent answer to the user's original question.

Original Question: "${state.originalQuestion}"

Company Briefs:
${briefs.map(b => `--- ${b.company} (years cited: ${b.yearsCited.join(", ") || "unknown"}) ---\n${b.text}`).join("\n\n")}
${correctionNote}

Instructions:
- Directly answer the original question, combining across companies.
- Preserve each brief's citations exactly as written — do not invent new ones.
${SCALE_PRESERVATION_RULE}
- Highlight the comparisons the user actually asked for.
- If briefs reference different fiscal years, say so explicitly rather than comparing as if same-period.
- At the end, list the companies and tools used as a short reference line.

Final Answer:`;

  const finalResponse = await draftingLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);

  return {
    finalOutput: finalResponse.content + renderSourceAppendix(state.evidence),
    agentTrace: [{ node: "finalizeMultiEntity" }],
  };
}

// ─────────────────────────────────────────────────────────────────────────
// ROUTING
// ─────────────────────────────────────────────────────────────────────────

function routeAfterClassifier(state) {
  const uniqueCompanies = [...new Set(state.subQuestions.flatMap(q => q.companies))];

  if (uniqueCompanies.length > 1) {
    const byCompany = {};
    for (const q of state.subQuestions) {
      const companies = q.companies.length ? q.companies : ["_general"];
      for (const c of companies) {
        byCompany[c] = byCompany[c] || [];
        byCompany[c].push({
          ...q,
          companies: [c],
          question: scopeQuestionToCompany(q.question, c, uniqueCompanies),
        });
      }
    }
    return uniqueCompanies.map((company) =>
      new Send("subAgent", {
        company,
        subQuestions: byCompany[company] || [],
        originalQuestion: state.originalQuestion,
        scopedQuestion: scopeQuestionToCompany(state.originalQuestion, company, uniqueCompanies),
        otherCompanies: uniqueCompanies.filter(c => c !== company),
        runStartTime: state.runStartTime,
      })
    );
  }

  const required = state.requiredTools;
  if (required.includes("RAG")) return "rag";
  if (required.includes("WEB")) return "web";
  if (required.includes("STOCK")) return "stock";
  return "synthesizer";
}

function routeAfterRag(state) {
  const required = state.requiredTools;
  const executed = state.toolsExecuted;
  if (required.includes("WEB") && !executed.includes("WEB")) return "web";
  if (required.includes("STOCK") && !executed.includes("STOCK")) return "stock";
  return "synthesizer";
}

function routeAfterWeb(state) {
  const required = state.requiredTools;
  const executed = state.toolsExecuted;
  if (required.includes("STOCK") && !executed.includes("STOCK")) return "stock";
  return "synthesizer";
}

function routeAfterVerifier(state) {
  const verdict = state.verifierVerdict?.verdict;
  if (verdict === "sufficient") return "__end__";
  if (verdict === "verification_unavailable") return "capReached";

  const isMultiEntityFinalize = (state.companyBriefs || []).length > 1;
  const maxIterations = isMultiEntityFinalize ? MAX_FINALIZE_VERIFIER_ITERATIONS : MAX_VERIFIER_ITERATIONS;

  if ((state.verifierAttempts || 0) >= maxIterations || budgetExceeded(state)) {
    return "capReached";
  }

  if (verdict === "needs_more_evidence") return "gapFill";
  if (verdict === "unsupported_claim" || verdict === "conflicting_evidence") return "resynthesize";
  return "__end__";
}

function routeAfterGapFill(state) {
  if ((state.companyBriefs || []).length > 1) return "finalizeMultiEntity";
  return "synthesizer";
}

// ─────────────────────────────────────────────────────────────────────────
// GRAPH
// ─────────────────────────────────────────────────────────────────────────

const graph = new StateGraph(AgentState)
  .addNode("classify", classifierNode)
  .addNode("rag", ragNode)
  .addNode("web", webNode)
  .addNode("stock", stockNode)
  .addNode("synthesizer", synthesizerNode)
  .addNode("verifier", verifierNode)
  .addNode("gapFill", gapFillNode)
  .addNode("resynthesize", resynthesizeNode)
  .addNode("capReached", capReachedNode)
  .addNode("subAgent", subAgentNode)
  .addNode("globalCheck", globalCheckNode)
  .addNode("finalizeMultiEntity", finalizeMultiEntityNode)

  .addEdge("__start__", "classify")
  .addConditionalEdges("classify", routeAfterClassifier, [
    "rag", "web", "stock", "synthesizer", "subAgent",
  ])
  .addConditionalEdges("rag", routeAfterRag)
  .addConditionalEdges("web", routeAfterWeb)
  .addEdge("stock", "synthesizer")
  .addEdge("synthesizer", "verifier")
  .addConditionalEdges("verifier", routeAfterVerifier)
  .addConditionalEdges("gapFill", routeAfterGapFill, ["synthesizer", "finalizeMultiEntity"])
  .addEdge("resynthesize", "verifier")
  .addEdge("capReached", "__end__")
  .addEdge("subAgent", "globalCheck")
  .addEdge("globalCheck", "finalizeMultiEntity")
  .addEdge("finalizeMultiEntity", "verifier")

  .compile();

export { graph };

// ─────────────────────────────────────────────────────────────────────────
// CLI harness
// ─────────────────────────────────────────────────────────────────────────

function printFullTrace(result) {
  console.log(chalk.blue.bold("\n───────────── FULL RUN TRACE ─────────────"));

  console.log(chalk.bold("\nSub-questions:"));
  (result.subQuestions || []).forEach((q, i) =>
    console.log(`  ${i + 1}. [${q.tool}] ${q.question}${q.companies?.length ? ` (${q.companies.join(", ")})` : ""}`)
  );

  console.log(chalk.bold("\nAgent trace (chronological):"));
  (result.agentTrace || []).forEach((t, i) => {
    console.log(`  ${i + 1}. [${t.node}]`, JSON.stringify(t, (k, v) => k === "node" ? undefined : v));
  });

  console.log(chalk.bold("\nEvidence collected:"));
  (result.evidence || []).forEach((e, i) => {
    console.log(`  ${i + 1}. [${e.tool}] Q: ${e.question}`);
    console.log(`     A: ${String(e.answer).slice(0, 200)}${e.answer?.length > 200 ? "…" : ""}`);
    console.log(`     sources: ${e.sources?.length ?? 0}`);
  });

  if (result.companyBriefs?.length) {
    console.log(chalk.bold("\nCompany briefs:"));
    result.companyBriefs.forEach(b => console.log(`  — ${b.company} (years: ${b.yearsCited.join(", ")})`));
    console.log(chalk.bold("Worst per-company retry count:"), result.subAgentMaxAttempts);
  }

  console.log(chalk.bold("\nFinal verifier verdict:"), JSON.stringify(result.verifierVerdict, null, 2));
  console.log(chalk.blue.bold("───────────────────────────────────────────\n"));
}

async function main() {
  const rl = readline.createInterface({ input, output });

  console.log(chalk.blue.bold("\n═══════════════════════════════════════════════════════"));
  console.log("  Investment Research AI - Agentic Multi-Hop RAG System");
  console.log(chalk.blue.bold("═══════════════════════════════════════════════════════"));
  console.log("Type your question or 'bye' to exit\n");

  while (true) {
    const userInput = await rl.question("You: ");
    if (userInput === "bye") break;

    const startTime = Date.now();
    const result = await graph.invoke({ originalQuestion: userInput });
    const endTime = Date.now();

    console.log(chalk.blue.bold("\n═══════════════════════════════════════════════════════\n"));
    console.log("Original Question:", result.originalQuestion);
    if (result.companyBriefs?.length > 1) {
      console.log("Mode: multi-entity —", result.companyBriefs.map(b => b.company).join(", "));
    } else {
      console.log("Mode: single-entity");
    }
    console.log("Final verdict:", result.verifierVerdict?.verdict, "| attempts:", result.verifierAttempts);

    console.log(chalk.blue.bold("\n═══════════════════════════════════════════════════════"));
    console.log(chalk.cyan(`  Execution time: ${((endTime - startTime) / 1000).toFixed(2)}s`));
    console.log(chalk.cyan(`  Tools used: ${result.toolsExecuted.join(" → ")}`));
    console.log(chalk.blue.bold("═══════════════════════════════════════════════════════\n"));

    console.log("\nAI:");
    console.log(marked(result.finalOutput));

    printFullTrace(result);
  }

  rl.close();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
import dotenv from "dotenv";
dotenv.config();
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { ChatGroq } from "@langchain/groq";
import { z } from "zod";
import { StateGraph } from "@langchain/langgraph";
import { CohereEmbeddings, CohereRerank } from "@langchain/cohere";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { marked } from "marked";
import TerminalRenderer from "marked-terminal";
import { TavilySearch } from "@langchain/tavily";
import { StockTool } from "./utils/stockTool.js";
import { routeInvestmentQuestion } from "./utils/routerModel.js";
import { resolveCik } from "./utils/secEdgar.js";
import chalk from 'chalk';

marked.setOptions({
  renderer: new TerminalRenderer(),
});

const supportState = z.object({
  originalQuestion: z.string(),

  subQuestions: z.array(
    z.object({
      question: z.string(),
      tool: z.enum(["RAG", "WEB", "STOCK"]),
      companies: z.array(z.string()),
      years: z.array(z.number()),
      searchQuery: z.string().optional(),
    }),
  ),

  requiredTools: z.array(z.enum(["RAG", "WEB", "STOCK"])),

  evidence: z
    .array(
      z.object({
        question: z.string(),
        tool: z.string(),
        answer: z.string(),
        sources: z.array(z.any()).optional(),
      }),
    )
    .optional(),

  toolsExecuted: z.array(z.string()).optional(),

  agentTrace: z.array(z.record(z.any())).optional(),

  finalOutput: z.string().optional(),
});

const smartLLM = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0.2,
  maxRetries: 2,
});

const fastLLM = new ChatGroq({
  model: "openai/gpt-oss-20b",
  temperature: 0.1,
  maxRetries: 2,
});

async function classifierNode(state) {
  const { subQuestions, requiredTools } = await routeInvestmentQuestion({
    question: state.originalQuestion,
  });

  console.log(
    "\nTo answer this question, the following tools will be used: ",
    requiredTools,
  );

  return {
    subQuestions,
    requiredTools,
    toolsExecuted: [],
    evidence: [],
    agentTrace: [],
  };
}

async function ragNode(state) {
  console.log("\n[RAG] Searching documents...\n");

  const embeddings = new CohereEmbeddings({
    model: "embed-english-v3.0",
    apiKey: process.env.COHERE_API_KEY,
  });

  const newTrace = [...(state.agentTrace || [])];
  const pinecone = new PineconeClient();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
  });

  // xbrl-fact chunks are few (≤9 per company/year) and already precise —
  // ground-truth numbers pulled straight from SEC's structured data, not
  // parsed HTML. They get their own retrieval lane: fetch everything
  // matching {company, year}, no vector-similarity competition, no rerank
  // cutoff. Relying on vector search for these was the actual bug — a
  // narrow top-K pool has no guarantee of surfacing a fact whose wording
  // doesn't happen to match the query, even for a simple bare-fact lookup
  // (confirmed: "What was Apple's revenue in FY2025?" failed to retrieve
  // the Revenues fact for exactly this reason).
  const MAX_XBRL_FACTS_PER_COMPANY_YEAR = 15; // effectively "all"

  // narrative/table-prose is the large, genuinely noisy pool — this is
  // where vector search + rerank actually earns its keep.
  const CANDIDATES_PER_COMPANY_YEAR = 12;
  const FINAL_NARRATIVE_CHUNKS_AFTER_RERANK = 6;

  const reranker = new CohereRerank({
    apiKey: process.env.COHERE_API_KEY,
    model: "rerank-english-v3.0",
    topN: FINAL_NARRATIVE_CHUNKS_AFTER_RERANK,
  });

  const newEvidence = [...(state.evidence || [])];

  // Presentation order only, applied after both lanes are merged.
  const SOURCE_TYPE_PRIORITY = { "xbrl-fact": 0, "table-prose": 1, narrative: 2 };

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "RAG") continue;

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

    // Lane 1: xbrl-fact — guaranteed, not vector-ranked
    const xbrlFactChunks = [];
    for (const company of companies) {
      for (const year of years) {
        const result = await vectorStore.similaritySearch(searchQuery, MAX_XBRL_FACTS_PER_COMPANY_YEAR, {
          company,
          year,
          sourceType: "xbrl-fact",
        });
        xbrlFactChunks.push(...result);
      }
    }

    const narrativeCandidates = [];
    for (const company of companies) {
      for (const year of years) {
        let result = await vectorStore.similaritySearch(searchQuery, CANDIDATES_PER_COMPANY_YEAR, { company, year });
        let narrative = result.filter((doc) => doc.metadata.sourceType !== "xbrl-fact");

        // Narrative/table-prose chunks are tagged with the filing's own year, not
        // every year they discuss — if we don't have that exact year ingested,
        // fall back to a company-only search rather than returning nothing. The
        // LLM sees each chunk's actual tagged year and can caveat accordingly.
        if (narrative.length === 0) {
          result = await vectorStore.similaritySearch(searchQuery, CANDIDATES_PER_COMPANY_YEAR, { company });
          narrative = result.filter((doc) => doc.metadata.sourceType !== "xbrl-fact");
        }

        narrativeCandidates.push(...narrative);
      }
    }

    console.log(
      chalk.green(`   ${xbrlFactChunks.length} xbrl-fact chunks (guaranteed), ${narrativeCandidates.length} narrative/table-prose candidates, reranking...\n`)
    );

    const rerankedNarrative =
      narrativeCandidates.length > 0
        ? await reranker.compressDocuments(narrativeCandidates, subQ.question)
        : [];

    const allChunks = [...xbrlFactChunks, ...rerankedNarrative].sort(
      (a, b) => SOURCE_TYPE_PRIORITY[a.metadata.sourceType] - SOURCE_TYPE_PRIORITY[b.metadata.sourceType]
    );

    newTrace.push({
      node: "rag",
      question: subQ.question,
      xbrlFactCount: xbrlFactChunks.length,
      narrativeCandidateCount: narrativeCandidates.length,
      rerankedCount: rerankedNarrative.length,
    });

    console.log(chalk.green(`   ${allChunks.length} total chunks\n`));

    const SYSTEM_PROMPT = `You are a financial research assistant.

                            Answer the question using ONLY the provided document excerpts.
                            Do NOT use outside knowledge.
                            Do NOT speculate.

                            Question: ${subQ.question}

                            Documents:
                            ${allChunks
                              .map(
                                (doc, i) => `
                            [${i + 1}] ${doc.metadata.company} ${doc.metadata.year} — ${doc.metadata.section} (${doc.metadata.sourceType}):
                            ${doc.pageContent}
                            `,
                              )
                              .join("\n")}

                            Instructions:
                            - Provide the factual answer.
                            - Include specific figures when available.
                            - If both an "xbrl-fact" and a "table-prose"/"narrative" excerpt give the same figure, prefer the xbrl-fact value — it comes from SEC's structured data, not parsed HTML.
                            - Cite sources using this format: [Company Year, Section].
                            - Do NOT add opinions or analysis beyond the documents.

                            Answer:
    `;

    const answerResponse = await fastLLM.invoke([{ role: "system", content: SYSTEM_PROMPT }]);

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
  }

  return {
    evidence: newEvidence,
    toolsExecuted: [...(state.toolsExecuted || []), "RAG"],
    agentTrace: newTrace,
  };
}

async function webNode(state) {
  console.log("\n[WEB] Searching web...");

  const newTrace = [...(state.agentTrace || [])];
  const webTool = new TavilySearch({
    maxResults: 5,
    topic: "news",
    searchDepth: "basic",
    includeAnswer: false,
  });

  const newEvidence = [...(state.evidence || [])];

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "WEB") continue;

    const query = subQ.companies?.length
      ? `${subQ.question} (${subQ.companies.join(", ")})`
      : subQ.question;

    const response = await webTool.invoke({ query });
    const searchResults = response.results.map((result) => result.content);

    newTrace.push({ node: "web", question: subQ.question, query: subQ.question, articlesFound: searchResults.length });

    console.log(chalk.green(`   Retrieved ${searchResults.length} articles\n`));

    const SYSTEM_PROMPT = `You are a financial research assistant.

                        Answer this question using ONLY the provided search results.

                        Question: ${subQ.question}

                        Search Results:
                        ${searchResults.map((result, i) => `[${i + 1}] ${result}`).join("\n\n")}

                        Provide a clear, factual answer. Do NOT use outside knowledge.

                        Answer:
    `;

    const answer = await fastLLM.invoke([
      { role: "system", content: SYSTEM_PROMPT },
    ]);

    newEvidence.push({
      question: subQ.question,
      tool: "WEB",
      answer: answer.content,
    });
  }
  return {
    evidence: newEvidence,
    toolsExecuted: [...(state.toolsExecuted || []), "WEB"],
    agentTrace: newTrace,
  };
}

async function stockNode(state) {
  console.log("\n[STOCK] Fetching market data...");

  const newTrace = [...(state.agentTrace || [])];
  const stockTool = new StockTool();
  const newEvidence = [...(state.evidence || [])];

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "STOCK") continue;

    try {
      const stockTrace = {};
      const stockData = await stockTool.answerQuestion(subQ.question, subQ.companies, stockTrace);
      newTrace.push({ node: "stock", question: subQ.question, ...stockTrace, resultCount: Array.isArray(stockData) ? stockData.length : 0 });

      if (!stockData || (Array.isArray(stockData) && stockData.length === 0)) {
        throw new Error("No data returned from API");
      }

      const formattedData = Array.isArray(stockData)
        ? stockData
            .map((d, idx) => {
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
            })
            .join("\n\n")
        : String(stockData);

      const SYSTEM_PROMPT = `You are a financial data analyst.

                            Answer this question using ONLY the provided stock market data.

                            Question: ${subQ.question}

                            Stock Data:
                            ${formattedData}

                            Provide a clear, factual answer with specific numbers. Compare data if multiple companies mentioned.

                            Answer:
      `;

      const answer = await fastLLM.invoke([
        { role: "system", content: SYSTEM_PROMPT },
      ]);

      newEvidence.push({
        question: subQ.question,
        tool: "STOCK",
        answer: answer.content,
        sources: Array.isArray(stockData)
          ? stockData.map((d) => ({
              ticker: d.ticker,
              price: d.price || d.endPrice,
              source: "Yahoo Finance",
            }))
          : [],
      });
    } catch (error) {
      console.log(`Error: ${error.message}\n`);

      newEvidence.push({
        question: subQ.question,
        tool: "STOCK",
        answer:
          "Unable to retrieve stock data. The ticker might be invalid or the service is temporarily unavailable.",
        sources: [],
      });
    }
  }

  return {
    evidence: newEvidence,
    toolsExecuted: [...(state.toolsExecuted || []), "STOCK"],
    agentTrace: newTrace,
  };
}

async function synthesizerNode(state) {
  console.log("\n[SYNTHESIZER] Combining evidence...\n");

  const SYSTEM_PROMPT = `You are a financial research analyst. Synthesize a comprehensive answer based on the evidence.

                        Original Question: "${state.originalQuestion}"

                        Evidence Collected:
                        ${(state.evidence || [])
                          .map(
                            (ev, i) => `
                        ${i + 1}. Sub-Question: ${ev.question}
                          Tool Used: ${ev.tool}
                          Answer: ${ev.answer}
                        `,
                          )
                          .join("\n")}

                        Instructions:
                      - Directly answer the user's question.
                      - Combine information across sub-questions logically.
                      - Cite sources using [Company Year, Page X].
                      - Highlight key comparisons or insights.
                      - Clearly state limitations or missing data if applicable.
                      - Do NOT introduce new facts not present in the evidence.
                      - At the end just mention the tools from which we have collected information to give this final answer. like a reference.

                        Final Answer:
  `;

  const finalResponse = await smartLLM.invoke([
    { role: "system", content: SYSTEM_PROMPT },
  ]);

  return {
    finalOutput: finalResponse.content,
  };
}

function routeAfterClassifier(state) {
  const required = state.requiredTools;

  if (required.includes("RAG")) return "rag";
  if (required.includes("WEB")) return "web";
  if (required.includes("STOCK")) return "stock"

  return "synthesizer";
}

function routeAfterRag(state) {
  const required = state.requiredTools;
  const executed = state.toolsExecuted;

  if (required.includes("WEB") && !executed.includes("WEB")) return "web";
  if (required.includes("STOCK") && !executed.includes("STOCK")) return 'stock'

  return "synthesizer";
}

function routeAfterWeb(state) {
  const required = state.requiredTools;
  const executed = state.toolsExecuted;

  if (required.includes("STOCK") && !executed.includes("STOCK")) return "stock";

  return "synthesizer";
}

const graph = new StateGraph(supportState)
  .addNode("classify", classifierNode)
  .addNode("rag", ragNode)
  .addNode("web", webNode)
  .addNode("stock", stockNode)
  .addNode("synthesizer", synthesizerNode)

  .addEdge("__start__", "classify")
  .addConditionalEdges("classify", routeAfterClassifier)
  .addConditionalEdges("rag", routeAfterRag)
  .addConditionalEdges("web", routeAfterWeb)
  .addEdge("stock", "synthesizer")
  .addEdge("synthesizer", "__end__")

  .compile();

export { graph }

// async function main() {
//   const rl = readline.createInterface({ input, output });

//   console.log(chalk.blue.bold('\n═══════════════════════════════════════════════════════'));
//   console.log("  Investment Research AI - Multi-Hop RAG System");
//   console.log(chalk.blue.bold('═══════════════════════════════════════════════════════'));
//   console.log("Type your question or 'bye' to exit\n");

//   while (true) {
//     const userInput = await rl.question("You: ");
//     if (userInput === "bye") break;

//     const startTime = Date.now();
//     const result = await graph.invoke({ originalQuestion: userInput });
//     const endTime = Date.now();

//     console.log(chalk.blue.bold('\n═══════════════════════════════════════════════════════\n'));

//     console.log("Original Question:", result.originalQuestion);
//     console.log("Sub-Questions:", result.subQuestions);
//     console.log("Evidence:", result.evidence);
    
//     console.log(chalk.blue.bold('\n═══════════════════════════════════════════════════════'));
//     console.log(chalk.cyan(`  Execution time: ${((endTime - startTime) / 1000).toFixed(2)}s`));
//     console.log(chalk.cyan(`  Tools used: ${result.toolsExecuted.join(' → ')}`));
//     console.log(chalk.blue.bold('═══════════════════════════════════════════════════════\n'));
  
//     console.log("\nAI:");
//     console.log(marked(result.finalOutput));
//   }

//   rl.close();
// }

// // Analyze NVIDIA vs AMD: data center growth, news sentiment, and stock performance
// // What is Microsoft's current stock price?

// main();
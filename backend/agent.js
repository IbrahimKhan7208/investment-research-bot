import dotenv from "dotenv";
dotenv.config();
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { ChatGroq } from "@langchain/groq";
import { z } from "zod";
import { StateGraph } from "@langchain/langgraph";
import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { marked } from "marked";
import TerminalRenderer from "marked-terminal";
import { TavilySearch } from "@langchain/tavily";
import { StockTool } from "./utils/stockTool.js";
import chalk from 'chalk';

marked.setOptions({
  renderer: new TerminalRenderer(),
});

const supportState = z.object({
  originalQuestion: z.string(),

  subQuestions: z.array(
    z.object({
      question: z.string(),
      tool: z.array(z.enum(["RAG", "WEB", "STOCK"])),
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

  finalOutput: z.string().optional(),
});

const smartLLM = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0.2,
  maxRetries: 2,
});

const fastLLM = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.1,
  maxRetries: 2,
});

async function extractFiltersFromQuestions(question) {
  try {
    const SYSTEM_PROMPT = `You extract structured search parameters from financial research questions.

                          Question: "${question}"

                          Available companies:
                          - NVIDIA
                          - AMD
                          - Microsoft

                          Available fiscal years:
                          - 2024
                          - 2025

                          Your task:
                          Extract which companies and years are relevant to the question,
                          and produce a cleaned search query suitable for semantic retrieval.

                          Rules:
                          - Company names must be exactly: "NVIDIA", "AMD", or "Microsoft"
                          - If multiple companies are mentioned, include all.
                          - If the question implies "recent", "latest", or "most recent", include both 2024 and 2025.
                          - Remove company names and years from the searchQuery.
                          - Do NOT infer information not present in the question.

                          Output format (STRICT JSON ONLY):
                          {
                            "companies": ["NVIDIA", "AMD", "Microsoft"],
                            "years": [2024, 2025],
                            "searchQuery": "string"
                          }
  `;

    const response = await smartLLM.invoke([
      { role: "system", content: SYSTEM_PROMPT },
    ]);

    const parsed = JSON.parse(response.content);

    return parsed;
  } catch {
    return {
      companies: ["NVIDIA", "AMD", "Microsoft"],
      years: [2024, 2025],
      searchQuery: question,
    };
  }
}

async function classifierNode(state) {
  const SYSTEM_PROMPT = `You are a planning component in an AI investment research system.

                        Your responsibility is to ANALYZE the user's question and PRODUCE A PLAN.
                        You do NOT answer questions.
                        You do NOT retrieve information.
                        You do NOT explain reasoning.

                        Your tasks:
                        1. Decompose the user's question into the MINIMUM number of factual sub-questions required to answer it accurately.
                        2. Assign EXACTLY ONE tool to EACH sub-question.

                        Available tools:
                        - RAG
                          Use when the answer can be found in internal documents only:
                          NVIDIA, AMD, Microsoft 10-K filings (2024, 2025)

                        - WEB
                          Use when the question requires information outside those documents, such as:
                          recent events
                          analyst opinions
                          news after filing dates

                        - STOCK
                          Use when the question involves:
                          stock price
                          market performance
                          market cap
                          trading volume

                        Rules:
                        - Each sub-question MUST use exactly ONE tool.
                        - If a sub-question would require multiple tools, SPLIT it.
                        - Sub-questions must be factual and specific (no explanations).
                        - Generate ONLY necessary sub-questions (do not over-split).
                        - Do NOT answer any sub-question.
                        - Do NOT add commentary or reasoning.

                        Output format (STRICT JSON ONLY):
                        {
                          "subQuestions": [
                            {
                              "question": "string",
                              "tool": "RAG" | "WEB" | "STOCK"
                            }
                          ],
                          "requiredTools": ["RAG", "WEB", "STOCK"]
                        }
  `;

  const response = await smartLLM.invoke([
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: state.originalQuestion },
  ]);

  const parsed = JSON.parse(response.content);
  console.log(
    "\nTo answer this question, the following tools will be used: ",
    parsed.requiredTools,
  );

  return {
    subQuestions: parsed.subQuestions,
    requiredTools: parsed.requiredTools,
    toolsExecuted: [],
    evidence: [],
  };
}

async function ragNode(state) {
  console.log("\n[RAG] Searching documents...\n");

  const embeddings = new CohereEmbeddings({
    model: "embed-english-v3.0",
    apiKey: process.env.COHERE_API_KEY,
  });

  const pinecone = new PineconeClient();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
  });

  const newEvidence = [...(state.evidence || [])];

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "RAG") continue;

    const filters = await extractFiltersFromQuestions(subQ.question);
    console.log(filters);

    const allChunks = [];

    for (const company of filters.companies) {
      for (const year of filters.years) {
        const result = await vectorStore.similaritySearch(
          filters.searchQuery,
          3,
          { company, year },
        );
        allChunks.push(...result);
      }
    }
    console.log(chalk.green(`   Found ${allChunks.length} relevant chunks\n`));

    const SYSTEM_PROMPT = `You are a financial research assistant.

                            Answer the question using ONLY the provided document excerpts.
                            Do NOT use outside knowledge.
                            Do NOT speculate.

                            Question: ${subQ.question}

                            Documents:
                            ${allChunks
                              .map(
                                (doc, i) => `
                            [${i + 1}] ${doc.metadata.company} ${doc.metadata.year} (Page ${doc.metadata.page}):
                            ${doc.pageContent}
                            `,
                              )
                              .join("\n")}

                            Instructions:
                            - Provide the factual answer.
                            - Include specific figures when available.
                            - Cite sources using this format: [Company Year, Page X].
                            - Do NOT add opinions or analysis beyond the documents.

                            Answer:
    `;

    const answerResponse = await fastLLM.invoke([
      { role: "system", content: SYSTEM_PROMPT },
    ]);

    // console.log(
    //   `Answer of ${allChunks[0].metadata.company}`,
    //   answerResponse.content,
    // );

    newEvidence.push({
      question: subQ.question,
      tool: "RAG",
      answer: answerResponse.content,
      sources: allChunks.map((doc) => ({
        company: doc.metadata.company,
        year: doc.metadata.year,
        page: doc.metadata.page,
      })),
    });
  }

  return {
    evidence: newEvidence,
    toolsExecuted: [...(state.toolsExecuted || []), "RAG"],
  };
}

async function webNode(state) {
  console.log("\n[WEB] Searching web...");

  const webTool = new TavilySearch({
    maxResults: 5,
    topic: "news",
    searchDepth: "basic",
    includeAnswer: false,
  });

  const newEvidence = [...(state.evidence || [])];

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "WEB") continue;

    const response = await webTool.invoke({ query: subQ.question });
    const searchResults = response.results.map((result) => result.content);

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
  };
}

async function stockNode(state) {
  console.log("\n[STOCK] Fetching market data...");

  const stockTool = new StockTool();
  const newEvidence = [...(state.evidence || [])];

  for (const subQ of state.subQuestions) {
    if (subQ.tool !== "STOCK") continue;

    try {
      const stockData = await stockTool.answerQuestion(subQ.question);

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
  };
}

function _formatStockData(stockData) {
  return stockData
    .map((data, idx) => {
      if (data.price) {
        // Current price data
        return `[${idx + 1}] ${data.ticker}
          Current Price: $${data.price?.toFixed(2)} ${data.currency}
          Change: ${data.change >= 0 ? "+" : ""}${data.change?.toFixed(2)} (${data.changePercent?.toFixed(2)}%)
          Volume: ${data.volume?.toLocaleString()}
          Market Cap: $${(data.marketCap / 1e9).toFixed(2)}B
          Previous Close: $${data.previousClose?.toFixed(2)}`;
      } else if (data.percentChange) {
        // Historical performance data
        return `[${idx + 1}] ${data.ticker} - ${data.period} Performance
          Start Price: $${data.startPrice?.toFixed(2)}
          End Price: $${data.endPrice?.toFixed(2)}
          Price Change: ${data.priceChange >= 0 ? "+" : ""}$${data.priceChange?.toFixed(2)}
          Percent Change: ${data.percentChange >= 0 ? "+" : ""}${data.percentChange}%
          Data Points: ${data.data.length} days`;
      }
    })
    .join("\n\n");
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
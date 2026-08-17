import "dotenv/config";
import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { resolveCik, getRecentFilings, buildFilingUrl, fetchFilingHtml } from "./utils/secEdgar.js";
import { filingHtmlToMarkdown } from "./utils/filingToMarkdown.js";
import { getFilingFacts } from "./utils/xbrlFacts.js";
import { chunkFiling } from "./utils/chunkFiling.js";

// Curated coverage list — offline ingestion, not auto-expanding (matches the
// "no ingestion SLA" scope: RAG covers whatever's been deliberately
// ingested, nothing fetched live per question).
//
// Using TICKERS, not company names — resolveCik checks exact ticker match
// first, before any name-substring logic, so this is deterministic. Names
// (e.g. "Google") can silently fail to resolve: SEC lists Alphabet as
// "Alphabet Inc.", and "google" isn't a substring of that title, so
// resolveCik's fallback matching would return null and skip it entirely.
//
// First 3 (NVDA/AMD/MSFT) match what was already validated end-to-end.
// Next 5 chosen to be in the router's training vocabulary (generate_dataset.py)
// so a RAG miss during testing can't be confused with a router extraction
// issue on an unfamiliar company, and to stress-test the parser against
// large, structurally complex filings. META is a deliberate test case for
// the router's known "Facebook" -> "Meta" alias gap.
const TICKERS = ["NVDA", "AMD", "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META"];

async function ingestCompany(vectorStore, ticker, filingsPerCompany = 3) {
  const company = await resolveCik(ticker);
  if (!company) {
    console.error(`Could not resolve CIK for "${ticker}", skipping`);
    return;
  }

  const filings = await getRecentFilings(company.cik, "10-K", filingsPerCompany);
  if (filings.length === 0) {
    console.error(`No 10-K found for ${ticker}, skipping`);
    return;
  }

  console.log(`Ingesting ${company.title} (${company.ticker}) — ${filings.length} filing(s)`);

  const BATCH_SIZE = 100;

  // XBRL facts are company-wide, not filing-specific — fetch/chunk ONCE.
  // Doing this per filing would duplicate every fact N times in the index.
  const facts = await getFilingFacts(company.cik, { formType: "10-K" });
  const factChunks = await chunkFiling("", facts, {
    company: company.ticker,
    formType: "10-K",
    filingDate: filings[0].filingDate,
  });
  console.log(`  ${factChunks.length} xbrl-fact chunks (company-wide history)`);
  for (let i = 0; i < factChunks.length; i += BATCH_SIZE) {
    await vectorStore.addDocuments(factChunks.slice(i, i + BATCH_SIZE));
  }

  // Narrative/table-prose IS filing-specific — each filing's actual MD&A
  // text only exists in that filing, so this has to run per filing to get
  // real multi-year coverage instead of just the latest year.
  for (const filing of filings) {
    console.log(`  Fetching ${filing.accessionNumber} (period ending ${filing.periodOfReport})`);
    const url = buildFilingUrl(company.cik, filing.accessionNumber, filing.primaryDocument);
    const html = await fetchFilingHtml(url);
    const markdown = filingHtmlToMarkdown(html);

    const baseMetadata = {
      company: company.ticker,
      year: new Date(filing.periodOfReport).getFullYear(),
      formType: "10-K",
      filingDate: filing.filingDate,
    };

    const narrativeChunks = await chunkFiling(markdown, [], baseMetadata); // [] — xbrl handled above, once
    console.log(`    ${narrativeChunks.length} narrative/table-prose chunks for FY${baseMetadata.year}`);
    for (let i = 0; i < narrativeChunks.length; i += BATCH_SIZE) {
      await vectorStore.addDocuments(narrativeChunks.slice(i, i + BATCH_SIZE));
    }
  }
}

async function main() {
  const embeddings = new CohereEmbeddings({
    model: "embed-english-v3.0",
    apiKey: process.env.COHERE_API_KEY,
  });
  const pinecone = new PineconeClient();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);
  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex });

  const results = { succeeded: [], failed: [] };

  for (const ticker of TICKERS) {
    try {
      await ingestCompany(vectorStore, ticker);
      results.succeeded.push(ticker);
    } catch (err) {
      console.error(`Failed on ${ticker}:`, err.message);
      results.failed.push({ ticker, reason: err.message });
    }
  }

  console.log("\n=== INGEST COMPLETE ===");
  console.log(`Succeeded: ${results.succeeded.length}/${TICKERS.length}`);
  if (results.failed.length > 0) {
    console.log("Failed:");
    results.failed.forEach((f) => console.log(`  - ${f.ticker}: ${f.reason}`));
  }
}

main().catch(console.error);
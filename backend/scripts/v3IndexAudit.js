import "dotenv/config";
import { CohereEmbeddings } from "@langchain/cohere";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";

const TICKERS = ["NVDA", "AMD", "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META"];
const MAX_PER_COMPANY = 2000; // safely above the largest company's chunk count seen so far (1088, META)

async function main() {
  const embeddings = new CohereEmbeddings({
    model: "embed-english-v3.0",
    apiKey: process.env.COHERE_API_KEY,
  });
  const pinecone = new PineconeClient();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

  // One embedding call, reused for every filtered pull below — since we're
  // fetching by metadata filter, not semantic relevance, the actual query
  // text is irrelevant. Pinecone's query API just requires *a* vector.
  const probeVector = await embeddings.embedQuery("financial report overview");

  const report = {};

  for (const ticker of TICKERS) {
    const result = await pineconeIndex.query({
      vector: probeVector,
      topK: MAX_PER_COMPANY,
      filter: { company: ticker },
      includeMetadata: true,
    });

    const matches = result.matches || [];
    const stats = {
      total: matches.length,
      bySourceType: {},
      byYear: {},
      bySection: {},
      issues: [],
      samples: {},
    };

    for (const m of matches) {
      const md = m.metadata || {};
      // LangChain's PineconeStore stores the chunk's pageContent under a
      // metadata key — default is "text". Falling back to "pageContent" in
      // case this project's setup used a custom textKey.
      const text = md.text ?? md.pageContent ?? "";

      stats.bySourceType[md.sourceType] = (stats.bySourceType[md.sourceType] || 0) + 1;
      stats.byYear[md.year] = (stats.byYear[md.year] || 0) + 1;
      stats.bySection[md.section] = (stats.bySection[md.section] || 0) + 1;

      if (!stats.samples[md.sourceType]) {
        stats.samples[md.sourceType] = {
          section: md.section,
          year: md.year,
          length: text.length,
          preview: text.slice(0, 200),
        };
      }

      // --- automated sanity checks ---
      if (!text.trim()) {
        stats.issues.push(`Empty pageContent — id ${m.id}`);
      }
      if (!md.company || md.year === undefined || !md.sourceType || !md.section) {
        stats.issues.push(`Missing metadata field — id ${m.id}: ${JSON.stringify(md)}`);
      }
      if (md.year !== undefined && (md.year < 2000 || md.year > new Date().getFullYear() + 1)) {
        stats.issues.push(`Implausible year ${md.year} — id ${m.id}, section ${md.section}`);
      }
      if (md.sourceType === "table-prose" && text.length > 1600) {
        stats.issues.push(`Oversized table-prose chunk (${text.length} chars) — id ${m.id}, section ${md.section}`);
      }
      if (md.sourceType === "narrative" && text.length > 1300) {
        stats.issues.push(`Oversized narrative chunk (${text.length} chars) — id ${m.id}, section ${md.section}`);
      }
    }

    if (matches.length >= MAX_PER_COMPANY) {
      stats.issues.push(
        `Hit topK cap (${MAX_PER_COMPANY}) — real count may be higher than shown, raise MAX_PER_COMPANY and re-run`
      );
    }

    report[ticker] = stats;
  }

  // ---- print ----
  let grandTotal = 0;
  for (const [ticker, stats] of Object.entries(report)) {
    grandTotal += stats.total;
    console.log(`\n${"=".repeat(60)}`);
    console.log(`${ticker} — ${stats.total} chunks in index`);
    console.log("=".repeat(60));
    console.log("By sourceType:", stats.bySourceType);
    console.log("By year:      ", stats.byYear);
    console.log("By section:   ", stats.bySection);

    console.log("\nSamples (one per sourceType):");
    for (const [type, sample] of Object.entries(stats.samples)) {
      console.log(`  [${type}] section=${sample.section} year=${sample.year} len=${sample.length}`);
      console.log(`    "${sample.preview}${sample.preview.length >= 200 ? "…" : ""}"`);
    }

    if (stats.issues.length > 0) {
      console.log(`\n  ${stats.issues.length} issue(s) found:`);
      stats.issues.slice(0, 10).forEach((i) => console.log("   -", i));
      if (stats.issues.length > 10) console.log(`   ...and ${stats.issues.length - 10} more`);
    } else {
      console.log("\n  No issues found.");
    }
  }

  console.log(`\n${"=".repeat(60)}`);
  console.log(`TOTAL across all ${TICKERS.length} companies: ${grandTotal} chunks`);
}

main().catch(console.error);
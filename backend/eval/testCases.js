// Test case shape:
//   id                — unique, used in reports
//   question          — sent verbatim to graph.invoke
//   tags              — for grouping/filtering (`node eval/runEval.js --tag=multi-entity`)
//   expectMultiEntity — does this question require >1 company (sub-agent fan-out)?
//   expectedSubstrings— case-insensitive strings the final answer should contain
//   knownLimitation   — if true, a fail here is tracked separately, not counted
//                        against the headline pass rate (documented gaps: alias
//                        canonicalization, relative-time year bias). A PASS on
//                        one of these is worth noticing — it means the gap closed.

export const testCases = [
  // ── Single-entity, single-tool ──────────────────────────────────────
  {
    id: "single-rag-basic-fact",
    question: "What was Apple's revenue in FY2025?",
    tags: ["single-entity", "rag", "smoke"],
    expectMultiEntity: false,
    expectedSubstrings: ["Apple"],
  },
  {
    id: "single-rag-derived-metric",
    question: "What is NVIDIA's operating margin for its most recent fiscal year?",
    tags: ["single-entity", "rag"],
    expectMultiEntity: false,
    expectedSubstrings: ["NVIDIA", "margin"],
  },
  {
    id: "single-stock-only",
    question: "What is Microsoft's current stock price?",
    tags: ["single-entity", "stock", "smoke"],
    expectMultiEntity: false,
    expectedSubstrings: ["Microsoft"],
  },
  {
    id: "single-web-only",
    question: "What is the current news sentiment on Amazon?",
    tags: ["single-entity", "web"],
    expectMultiEntity: false,
    expectedSubstrings: ["Amazon"],
  },

  // ── Single-entity, multi-tool (exercises full RAG+WEB+STOCK path) ──
  {
    id: "single-multitool-tesla",
    question: "Analyze Tesla: recent financials, news sentiment, and stock performance",
    tags: ["single-entity", "multi-tool"],
    expectMultiEntity: false,
    expectedSubstrings: ["Tesla", "revenue"],
  },
  {
    id: "single-multitool-google",
    question: "Give me a full analysis of Alphabet: financials, recent news, and stock trend",
    tags: ["single-entity", "multi-tool"],
    expectMultiEntity: false,
    expectedSubstrings: ["Alphabet"],
  },

  // ── Multi-entity ─────────────────────────────────────────────────────
  {
    id: "multi-two-company-comparison",
    question: "Compare NVIDIA and AMD's data center revenue growth",
    tags: ["multi-entity", "comparison"],
    expectMultiEntity: true,
    expectedSubstrings: ["NVIDIA", "AMD"],
  },
  {
    id: "multi-unrelated-asks",
    question: "What is NVIDIA's revenue, and what is Tesla's biggest risk factor?",
    tags: ["multi-entity", "dispatch-trigger"],
    expectMultiEntity: true,
    expectedSubstrings: ["NVIDIA", "Tesla"],
  },
  {
    id: "multi-three-company",
    question: "Compare the operating margins of Microsoft, Apple, and Amazon",
    tags: ["multi-entity", "stress", "budget"],
    expectMultiEntity: true,
    expectedSubstrings: ["Microsoft", "Apple", "Amazon"],
  },
  {
    id: "multi-full-analysis-two-company",
    question: "Compare Tesla and AMD's recent financials, news sentiment, and stock performance",
    tags: ["multi-entity", "multi-tool", "stress"],
    expectMultiEntity: true,
    expectedSubstrings: ["Tesla", "AMD"],
  },
  {
    id: "multi-mismatched-year-risk",
    question: "Compare Meta and NVIDIA's most recent revenue growth",
    tags: ["multi-entity", "coherence-check"],
    expectMultiEntity: true,
    expectedSubstrings: ["Meta", "NVIDIA"],
    // META has fewer ingested filings than NVDA (see README Data Coverage) —
    // good candidate for surfacing a real year-mismatch flag in globalCheck.
  },

  // ── Known limitations — expected to fail until root-caused ──────────
  {
    id: "known-limitation-alias-canonicalization",
    question: "What was Facebook's revenue last year?",
    tags: ["known-limitation", "router"],
    expectMultiEntity: false,
    expectedSubstrings: ["Meta"],
    knownLimitation: true,
  },
  {
    id: "known-limitation-relative-time",
    question: "What is NVIDIA's most recent quarterly revenue trend?",
    tags: ["known-limitation", "router"],
    expectMultiEntity: false,
    expectedSubstrings: ["NVIDIA"],
    knownLimitation: true,
  },

  // ── Thin-coverage / gap-fill triggers ────────────────────────────────
  {
    id: "gapfill-older-year-narrative",
    question: "What did AMD say about its data center growth strategy in FY2024?",
    tags: ["gap-fill", "rag"],
    expectMultiEntity: false,
    expectedSubstrings: ["AMD"],
  },

  // ── Edge / boundary cases ────────────────────────────────────────────
  {
    id: "edge-vague-question",
    question: "How is Apple doing?",
    tags: ["edge"],
    expectMultiEntity: false,
    expectedSubstrings: ["Apple"],
  },
  {
    id: "edge-explicit-old-year",
    question: "What was Amazon's net income in FY2023?",
    tags: ["edge", "rag"],
    expectMultiEntity: false,
    expectedSubstrings: ["Amazon"],
  },
];
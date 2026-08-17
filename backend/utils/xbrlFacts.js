const USER_AGENT = process.env.SEC_USER_AGENT;

async function fetchCompanyFacts(cik) {
  const res = await fetch(`https://data.sec.gov/api/xbrl/companyfacts/CIK${cik}.json`, {
    headers: { "User-Agent": USER_AGENT },
  });
  if (!res.ok) throw new Error(`Failed to fetch company facts for CIK ${cik}: ${res.status}`);
  return res.json();
}

// Core us-gaap concepts covering the primary Item 8 statements. Companies
// sometimes use alternate tags for the same concept (e.g. Revenues vs.
// RevenueFromContractWithCustomerExcludingAssessedTax) - listed in
// priority order, first match wins. Not exhaustive; expand as needed.
const CORE_CONCEPTS = {
  Assets: ["Assets"],
  Liabilities: ["Liabilities"],
  StockholdersEquity: ["StockholdersEquity"],
  CashAndCashEquivalents: ["CashAndCashEquivalentsAtCarryingValue"],
  Revenues: ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
  NetIncomeLoss: ["NetIncomeLoss"],
  GrossProfit: ["GrossProfit"],
  OperatingIncomeLoss: ["OperatingIncomeLoss"],
  ResearchAndDevelopmentExpense: ["ResearchAndDevelopmentExpense"],
};

function formatValue(val, unit) {
  if (unit === "USD") {
    const millions = val / 1_000_000;
    return `$${millions.toLocaleString("en-US", { maximumFractionDigits: 1 })} million`;
  }
  return `${val} ${unit}`;
}

export async function getFilingFacts(cik, { formType = "10-K", fiscalYear } = {}) {
  const facts = await fetchCompanyFacts(cik);
  const usGaap = facts.facts["us-gaap"] || {};

  // Collect matches deduped by (concept, fiscal year). SEC's companyfacts
  // repeats each comparative figure across every filing that discloses it
  // (e.g. FY2024 revenue appears as the "current" figure in the FY2024 10-K
  // AND as a comparative in the FY2025 10-K) — both entries carry the same
  // real-world value for the same fiscal year, just filed at different
  // times. Without dedup, calling this without a fiscalYear filter (to pull
  // one company's full history in one ingest pass) would index the same
  // fact multiple times. Keep only the most recently filed disclosure per
  // (concept, fiscal year).
  const bestByKey = new Map();

  for (const [conceptName, tagCandidates] of Object.entries(CORE_CONCEPTS)) {
    const tag = tagCandidates.find((t) => usGaap[t]);
    if (!tag) continue;

    const concept = usGaap[tag];
    for (const [unit, entries] of Object.entries(concept.units)) {
      for (const e of entries) {
        if (e.form !== formType || e.fp !== "FY") continue;

        // e.fy is the fiscal year of the FILING the fact was disclosed in,
        // not the fiscal year the value is actually about — a 10-K reports
        // 2-3 years of comparatives, and SEC tags all of them with the
        // filing's own fy. Derive the real fiscal year from the period-end
        // date instead, matching the convention used everywhere else in
        // this project (e.g. NVIDIA FY2025 = period ending Jan 2025).
        const derivedFY = new Date(e.end).getFullYear();
        if (fiscalYear && derivedFY !== fiscalYear) continue;

        const key = `${conceptName}-${derivedFY}`;
        const existing = bestByKey.get(key);
        if (!existing || e.filed > existing.entry.filed) {
          bestByKey.set(key, { entry: e, unit, conceptName, concept, derivedFY });
        }
      }
    }
  }

  const sentences = [];
  for (const { entry: m, unit, conceptName, concept, derivedFY } of bestByKey.values()) {
    sentences.push({
      concept: conceptName,
      fiscalYear: derivedFY,
      periodEnd: m.end,
      text: `${conceptName} (${concept.label}) for fiscal year ${derivedFY}, period ending ${m.end}: ${formatValue(m.val, unit)}. Source: ${formType} filed ${m.filed}.`,
    });
  }

  return sentences;
}
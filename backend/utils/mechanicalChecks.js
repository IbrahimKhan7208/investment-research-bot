// Cheap, code-only checks that run BEFORE the global LLM verifier call.
// Catches things an LLM would have to read carefully to notice — a mismatch
// in cited fiscal years across briefs is exact-match logic, not judgment.

export function setsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

export function mechanicalCoherenceCheck(briefs) {
  const flags = [];

  if (briefs.length > 1) {
    const yearSets = briefs.map(b => new Set(b.yearsCited));
    const allSame = yearSets.every(s => setsEqual(s, yearSets[0]));
    if (!allSame) {
      flags.push({
        type: "year_mismatch",
        detail: briefs.map(
          b => `${b.company}: FY${[...b.yearsCited].join(",") || "unknown"}`
        ),
      });
    }
  }

  return flags;
}
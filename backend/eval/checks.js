// ─────────────────────────────────────────────────────────────────────────
// Regression checks tied to specific real bugs found during development,
// not generic quality heuristics. Each one exists because something
// concrete broke this way once.
// ─────────────────────────────────────────────────────────────────────────

// The RAG extraction prompt now explicitly forbids 【N】-style footnote
// markers (models drifted toward them instead of the required
// [Company Year, Section] format). This is a hard regression check — the
// format is instructed unconditionally, so any occurrence is a real fail,
// not a heuristic guess.
export function hasStrayCitationMarkers(text) {
  return /【[^】]*】/.test(text || "");
}

const SCALE_MULTIPLIERS = {
  thousand: 1e3, k: 1e3,
  million: 1e6, m: 1e6,
  billion: 1e9, b: 1e9,
};

// Pulls every "$X,XXX million" / "$X.XB" / "3.5M" style figure out of a
// block of text and normalizes it to a raw number for comparison.
export function extractScaledNumbers(text) {
  const regex = /\$?\s?(\d[\d,]*(?:\.\d+)?)\s?(thousand|million|billion|[kmb])\b/gi;
  const out = [];
  let match;
  while ((match = regex.exec(text || "")) !== null) {
    const num = parseFloat(match[1].replace(/,/g, ""));
    const unit = match[2].toLowerCase();
    const mult = SCALE_MULTIPLIERS[unit] ?? SCALE_MULTIPLIERS[unit[0]] ?? 1;
    if (!isNaN(num)) out.push({ value: num * mult, raw: match[0].trim() });
  }
  return out;
}

// HEURISTIC, not a hard pass/fail — this is intentionally lenient and
// reported as warnings, never failures. A merged answer legitimately
// paraphrases, rounds, or omits some of a brief's numbers; that's not a
// bug. What IS worth flagging: a brief citing a real figure that reappears
// in the final answer at a magnitude that's off by an exact power of 1000
// (million vs billion, etc) — that specific shape is what the real AMD
// "$3,694 million" -> "$3.694 M" bug looked like. Exact-value drift within
// noise (rounding, currency formatting) is not what this checks for.
export function checkMagnitudeConsistency(companyBriefs, finalOutput) {
  const warnings = [];
  const finalNums = extractScaledNumbers(finalOutput);

  for (const brief of companyBriefs || []) {
    const briefNums = extractScaledNumbers(brief.text);
    for (const bn of briefNums) {
      const exactMatch = finalNums.some(fn => Math.abs(fn.value - bn.value) / bn.value < 0.01);
      if (exactMatch) continue;

      // Specifically check for a 1000x (or 1/1000x) drift — the exact
      // signature of a million/billion unit-scale corruption during merge.
      const scaleDrift = finalNums.some(fn => {
        const ratio = fn.value / bn.value;
        return Math.abs(ratio - 1000) < 1 || Math.abs(ratio - 0.001) < 0.000001;
      });

      if (scaleDrift) {
        warnings.push(`${brief.company}: "${bn.raw}" appears scale-corrupted (~1000x drift) in the final answer`);
      }
    }
  }
  return warnings;
}
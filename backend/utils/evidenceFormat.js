// utils/evidenceFormat.js

export function buildEvidenceBlock(evidence) {
  let webCounter = 0; // global across the whole evidence set, not per sub-question —
                       // keeps [W#] tags unique so they never collide once merged.

  return (evidence || []).map((ev, i) => {
    let sourcesText;

    if (!ev.sources || ev.sources.length === 0) {
      sourcesText = "(no structured sources attached)";
    } else if (ev.tool === "RAG") {
      sourcesText = ev.sources.map((s) => `  - [${s.company} ${s.year}, ${s.section}]`).join("\n");
    } else if (ev.tool === "WEB") {
      sourcesText = ev.sources.map((s) => {
        webCounter += 1;
        return `  - [W${webCounter}] ${s.title}`;
      }).join("\n");
    } else if (ev.tool === "STOCK") {
      sourcesText = ev.sources.map((s) => `  - [${s.ticker}, Yahoo Finance] price: $${s.price}`).join("\n");
    } else {
      sourcesText = "(unrecognized source format)";
    }

    return `
${i + 1}. Sub-Question: ${ev.question}
  Tool Used: ${ev.tool}
  Answer: ${ev.answer}
  Available sources for this sub-question:
${sourcesText}
`;
  }).join("\n");
}

export const CITATION_RULES = `- RAG sources: [Company Year, Section] — e.g. [NVDA 2026, Item 7]
- WEB sources: cite using the exact [W#] tag shown next to that source in "Available sources" — never write a raw URL yourself, never invent a number.
- STOCK sources: [Ticker, Yahoo Finance]
- Only cite sources actually listed in the evidence — never invent a citation.
- If evidence text contains a citation marker that doesn't match one of the exact formats above (e.g. bracket footnotes like 【8】, bare numbers like [1] with no matching WEB tag), do NOT reinterpret or translate it into one of the formats above — omit that citation rather than guessing what it refers to.`;

export const SCALE_PRESERVATION_RULE = `- Preserve every figure's exact scale (millions vs billions vs raw units) exactly as stated in its source — do not reformat, abbreviate, or convert a number in a way that changes its magnitude.`;
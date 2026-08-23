import { z } from "zod";
import { buildEvidenceBlock } from "./evidenceFormat.js";

export const verifierVerdictSchema = z.object({
  verdict: z.enum([
    "sufficient",
    "needs_more_evidence",
    "unsupported_claim",
    "conflicting_evidence",
  ]),
  reasoning: z.string(),

  claimsChecked: z.array(z.object({
    claim: z.string(),
    supported: z.boolean(),
    citedAs: z.string().optional(),
  })).optional(),

  evidenceGaps: z.array(z.object({
    missing: z.string(),
    suggestedTool: z.enum(["RAG", "WEB", "STOCK"]),
    suggestedQuery: z.string(),
    companies: z.array(z.string()).optional(),
    years: z.array(z.number()).optional(),
  })).optional(),

  conflicts: z.array(z.object({
    claimA: z.string(), sourceA: z.string(),
    claimB: z.string(), sourceB: z.string(),
  })).optional(),
}).refine(v => {
  if (v.verdict === "needs_more_evidence") return v.evidenceGaps?.length > 0;
  if (v.verdict === "unsupported_claim") return v.claimsChecked?.some(c => !c.supported);
  if (v.verdict === "conflicting_evidence") return v.conflicts?.length > 0;
  return true;
}, { message: "verdict requires its corresponding evidence array to be populated" });

export function buildVerifierPrompt(params) {
  return params.mode === "global" ? buildGlobalPrompt(params) : buildGroundingPrompt(params);
}

function buildGroundingPrompt({ mode, question, company, evidenceBundle, draftText }) {
  const scopeLine = mode === "local"
    ? `Company: ${company}\nSub-question(s) this brief answers: ${question}`
    : `Original question: ${question}`;
  const noun = mode === "local" ? "brief" : "answer";

  return `You are a verification agent checking a financial research ${noun} before it is ${mode === "local" ? "returned upstream" : "shown to the user"}.

${scopeLine}

Evidence available to the ${noun}'s author:
${buildEvidenceBlock(evidenceBundle)}

Draft ${noun}:
"""
${draftText}
"""

Check the ${noun} against the evidence ONLY — do not use outside knowledge of any company or financial markets.

For every factual claim in the ${noun} (a number, a trend, a comparison, an attribution):
1. Does it appear in the evidence above, or follow directly from it?
2. If it cites a source, does that source actually say what's claimed?
3. If two pieces of evidence disagree on the same fact, does the ${noun} acknowledge that, or silently pick one?

Return exactly one verdict:
- "sufficient": every claim is traceable to evidence, no unresolved conflicts.
- "needs_more_evidence": accurate as far as it goes, but doesn't fully answer the question because the evidence is incomplete — not because the ${noun}'s author reasoned poorly.
- "unsupported_claim": states something evidence doesn't support (a number not in any excerpt, a cited source that doesn't say what's cited, or a comparison/inference invented beyond what evidence shows).
- "conflicting_evidence": the evidence itself disagrees on a material fact and the ${noun} doesn't flag it.

If evidence disagrees AND there's also an unsupported claim, prefer "unsupported_claim".

When suggesting evidenceGaps, include which companies/years each gap concerns if determinable from the question — this feeds a re-retrieval step directly, don't leave it for re-parsing.

Do not judge writing quality or completeness of prose — only factual traceability. You are checking the evidence-vs-draft link only; you have no way to tell if the original question was mis-decomposed or routed to the wrong tool upstream — don't try to second-guess that here.`;
}

function buildGlobalPrompt({ originalQuestion, briefs, mechanicalFlags }) {
  return `You are checking a set of already locally-verified company briefs for cross-entity coherence before final synthesis.

Original question: ${originalQuestion}

Briefs (each already individually verified against its own evidence — do NOT re-check individual facts, only cross-brief coherence):
${briefs.map(b => `— ${b.company} (years cited: ${b.yearsCited.join(", ")}):\n${b.text}`).join("\n\n")}

Automated pre-checks flagged the following — confirm, dismiss, or reclassify each based on whether it actually affects the answer:
${mechanicalFlags.length ? mechanicalFlags.map(f => `- ${f.type}: ${JSON.stringify(f.detail)}`).join("\n") : "(none)"}

Also check for:
- Two briefs stating incompatible things about the same fact
- A brief silently omitting a company the original question asked about

Return the same verdict schema. "conflicting_evidence" here means brief-vs-brief, not evidence-vs-evidence within one brief — that's already resolved locally.`;
}

export async function verifyGrounding({ llm, ...promptParams }) {
  const prompt = buildVerifierPrompt(promptParams);
  const structuredLLM = llm.withStructuredOutput(verifierVerdictSchema);
  return structuredLLM.invoke([{ role: "system", content: prompt }, { role: "user", content: "Proceed." }]);
}
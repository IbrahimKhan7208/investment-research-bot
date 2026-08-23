// utils/verifierGuard.js
import { verifyGrounding } from "./verifier.js";

// Run-wide, across EVERY verifyGrounding call in one graph.invoke() —
// every sub-agent's local loop, the global-coherence check, and the
// top-level (single-entity / post-finalize) loop, combined. Free-tier
// Gemini quota is 20 requests/DAY total, not per run — a single
// multi-entity question can otherwise spend 10+ calls across three
// independently-capped loops that don't know about each other, leaving
// no headroom for a second test the same day. This is the shared ceiling
// that ties them together.
export const MAX_VERIFIER_CALLS_PER_RUN = 15;

let callsThisRun = 0;

// Called once per new question, before graph.invoke() — see main() below.
export function resetVerifierBudget() {
  callsThisRun = 0;
}

export function verifierCallsSoFar() {
  return callsThisRun;
}

// Every call site uses this instead of calling verifyGrounding directly.
// Two failure modes, same handling: budget exhausted (skip the call
// entirely — don't even try) and a live provider error (attempt it, catch
// failure). Either way the caller gets back a verdict-shaped object it can
// treat as terminal, never an uncaught exception that kills the process.
export async function guardedVerify(params, label = "verify") {
  if (callsThisRun >= MAX_VERIFIER_CALLS_PER_RUN) {
    console.warn(`   [${label}] run-wide verifier budget (${MAX_VERIFIER_CALLS_PER_RUN}) reached — skipping call.`);
    return {
      verdict: "verification_unavailable",
      reasoning: `Run-wide verifier call budget (${MAX_VERIFIER_CALLS_PER_RUN}) was reached before this check could run. Treating the current draft as best-effort rather than continuing to retry.`,
      claimsChecked: [],
      evidenceGaps: [],
      conflicts: [],
    };
  }

  callsThisRun += 1;
  try {
    return await verifyGrounding(params);
  } catch (err) {
    console.warn(`   [${label}] verifier call failed: ${err.message}`);
    return {
      verdict: "verification_unavailable",
      reasoning: `The verifier call itself failed (${err.message}) — likely a rate limit or transient provider error. Treating the current draft as best-effort rather than retrying, since retrying immediately is likely to hit the same failure again.`,
      claimsChecked: [],
      evidenceGaps: [],
      conflicts: [],
    };
  }
}
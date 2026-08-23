// Round 1: use each gap's own suggestedTool. Round 2+: force WEB, unless
// WEB was already the tool tried — in that case there's nowhere further to
// escalate to, so the gap is abandoned rather than retried a third
// identical way. No round 3 for gaps: by round 2's outcome you're either
// resolved or the escalation path is already exhausted.
export function planGapRetrieval(gaps, roundNumber) {
  return gaps.map((g) => {
    if (roundNumber <= 1) {
      return { ...g, tool: g.suggestedTool, escalated: false, giveUp: false };
    }
    if (g.suggestedTool === "WEB") {
      return { ...g, tool: null, escalated: false, giveUp: true };
    }
    return { ...g, tool: "WEB", escalated: true, giveUp: false, originalTool: g.suggestedTool };
  });
}
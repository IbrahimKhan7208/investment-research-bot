export const MAX_VERIFIER_ITERATIONS = 3;
export const MAX_FINALIZE_VERIFIER_ITERATIONS = 1;
export const MAX_TOOL_CALLS_PER_RUN = 20;
export const MAX_RUN_ELAPSED_MS = 10 * 60 * 1000;

export function computeToolCallCount(agentTrace) {
  return (agentTrace || []).filter(t => ["rag", "web", "stock"].includes(t.node)).length;
}

export function budgetExceeded(state) {
  const toolCalls = computeToolCallCount(state.agentTrace);
  const elapsed = Date.now() - (state.runStartTime || Date.now());
  return toolCalls >= MAX_TOOL_CALLS_PER_RUN || elapsed >= MAX_RUN_ELAPSED_MS;
}
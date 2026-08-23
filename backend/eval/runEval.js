import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import chalk from "chalk";

import { graph } from "../agent.js";
import { testCases } from "./testCases.js";
import { hasStrayCitationMarkers, checkMagnitudeConsistency } from "./checks.js";
import { computeToolCallCount } from "../utils/loopControl.js";
import { MAX_VERIFIER_CALLS_PER_RUN, verifierCallsSoFar } from "../utils/verifierGuard.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RESULTS_DIR = path.join(__dirname, "results");

// Optional: `node eval/runEval.js --tag=multi-entity` runs only matching cases.
const tagFilter = process.argv.find(a => a.startsWith("--tag="))?.split("=")[1];
const casesToRun = tagFilter
  ? testCases.filter(tc => tc.tags?.includes(tagFilter))
  : testCases;

function scoreCase(tc, result, elapsedMs) {
  const notes = [];
  let pass = true;

  const wasMultiEntity = (result.companyBriefs || []).length > 1;
  if (wasMultiEntity !== tc.expectMultiEntity) {
    pass = false;
    notes.push(`entity-mode mismatch: expected multi=${tc.expectMultiEntity}, got multi=${wasMultiEntity}`);
  }

  for (const s of tc.expectedSubstrings || []) {
    if (!result.finalOutput?.toLowerCase().includes(s.toLowerCase())) {
      pass = false;
      notes.push(`missing expected mention: "${s}"`);
    }
  }

  if (hasStrayCitationMarkers(result.finalOutput)) {
    pass = false;
    notes.push('found 【N】-style citation marker — the explicit prohibition in ragNode\'s prompt regressed');
  }

  const magnitudeWarnings = checkMagnitudeConsistency(result.companyBriefs, result.finalOutput);
  magnitudeWarnings.forEach(w => notes.push(`[heuristic, non-blocking] ${w}`));

  const finalVerdict = result.verifierVerdict?.verdict;
  if (finalVerdict === "cap_reached") notes.push("hit the verification retry cap — review manually, not an automatic fail");
  if (finalVerdict === "verification_unavailable") notes.push("verifier was unavailable (budget or provider error) — review manually");

  return {
    id: tc.id,
    tags: tc.tags || [],
    knownLimitation: !!tc.knownLimitation,
    pass,
    notes,
    mode: wasMultiEntity ? "multi" : "single",
    verdict: finalVerdict,
    verifierAttempts: result.verifierAttempts,
    subAgentMaxAttempts: result.subAgentMaxAttempts,
    toolCallCount: computeToolCallCount(result.agentTrace),
    verifierCallsUsed: verifierCallsSoFar(),
    elapsedMs,
  };
}

function printSummary(results) {
  console.log(chalk.blue.bold("\n═══════════════════════════════════════════════════════"));
  console.log(chalk.blue.bold("  EVAL SUMMARY"));
  console.log(chalk.blue.bold("═══════════════════════════════════════════════════════\n"));

  let passCount = 0, expectedFailCount = 0, realFailCount = 0;

  for (const r of results) {
    const label = r.threw ? chalk.red("ERROR")
      : r.pass ? chalk.green("PASS")
      : r.knownLimitation ? chalk.yellow("EXPECTED-FAIL")
      : chalk.red("FAIL");

    console.log(
      `${label}  ${r.id}  ` +
      chalk.gray(`(${(r.elapsedMs / 1000).toFixed(1)}s, mode=${r.mode}, verdict=${r.verdict}, ` +
        `verifierAttempts=${r.verifierAttempts}, subAgentMaxAttempts=${r.subAgentMaxAttempts ?? "-"}, ` +
        `toolCalls=${r.toolCallCount}, verifierCalls=${r.verifierCallsUsed}/${MAX_VERIFIER_CALLS_PER_RUN})`)
    );
    for (const n of r.notes || []) console.log(chalk.gray(`   - ${n}`));

    if (r.threw) continue;
    if (r.pass) passCount++;
    else if (r.knownLimitation) expectedFailCount++;
    else realFailCount++;
  }

  const threwCount = results.filter(r => r.threw).length;

  console.log(chalk.blue.bold("\n───────────────────────────────────────────────────────"));
  console.log(`Passed:            ${passCount}/${results.length}`);
  console.log(`Expected fails:    ${expectedFailCount} (known limitations — a pass here means a gap closed)`);
  console.log(chalk.red.bold(`Unexpected fails:  ${realFailCount}`));
  if (threwCount > 0) console.log(chalk.red.bold(`Threw / crashed:   ${threwCount}`));

  const avgElapsed = results.reduce((s, r) => s + (r.elapsedMs || 0), 0) / results.length;
  const avgVerifierCalls = results.reduce((s, r) => s + (r.verifierCallsUsed || 0), 0) / results.length;
  console.log(`Avg latency:       ${(avgElapsed / 1000).toFixed(1)}s`);
  console.log(`Avg verifier calls/run: ${avgVerifierCalls.toFixed(1)} (budget ${MAX_VERIFIER_CALLS_PER_RUN})`);
  console.log(chalk.blue.bold("═══════════════════════════════════════════════════════\n"));
}

function writeReports(results, meta) {
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");

  const jsonPath = path.join(RESULTS_DIR, `${timestamp}.json`);
  fs.writeFileSync(jsonPath, JSON.stringify({ meta, results }, null, 2));

  // "latest.json" is the file a future run diffs against, kept separate
  // from the timestamped archive so regression comparison doesn't need
  // to know the previous run's filename.
  fs.writeFileSync(path.join(RESULTS_DIR, "latest.json"), JSON.stringify({ meta, results }, null, 2));

  const passCount = results.filter(r => r.pass && !r.threw).length;
  const md = [
    `# Eval Report — ${meta.timestamp}`,
    "",
    `${passCount}/${results.length} passed · ${results.filter(r => r.knownLimitation).length} known-limitation cases · ${results.filter(r => r.threw).length} crashed`,
    "",
    "| ID | Result | Mode | Verdict | Attempts | Tool Calls | Verifier Calls | Latency |",
    "|---|---|---|---|---|---|---|---|",
    ...results.map(r =>
      `| ${r.id} | ${r.threw ? "ERROR" : r.pass ? "PASS" : r.knownLimitation ? "EXPECTED-FAIL" : "**FAIL**"} | ${r.mode ?? "-"} | ${r.verdict ?? "-"} | ${r.verifierAttempts ?? "-"} | ${r.toolCallCount ?? "-"} | ${r.verifierCallsUsed ?? "-"} | ${r.elapsedMs ? (r.elapsedMs / 1000).toFixed(1) + "s" : "-"} |`
    ),
    "",
    "## Notes",
    ...results.flatMap(r => (r.notes || []).map(n => `- **${r.id}**: ${n}`)),
  ].join("\n");

  fs.writeFileSync(path.join(RESULTS_DIR, "latest.md"), md);
  return jsonPath;
}

async function main() {
  console.log(chalk.blue.bold(`\nRunning ${casesToRun.length} eval case(s)${tagFilter ? ` (tag: ${tagFilter})` : ""}...\n`));

  const results = [];

  for (const tc of casesToRun) {
    console.log(chalk.cyan(`\n▶ ${tc.id}`));
    const start = Date.now();
    try {
      const result = await graph.invoke({ originalQuestion: tc.question });
      const elapsedMs = Date.now() - start;
      results.push(scoreCase(tc, result, elapsedMs));
    } catch (error) {
      const elapsedMs = Date.now() - start;
      results.push({
        id: tc.id,
        tags: tc.tags || [],
        knownLimitation: !!tc.knownLimitation,
        pass: false,
        threw: true,
        notes: [`threw: ${error.message}`],
        elapsedMs,
      });
    }
  }

  printSummary(results);

  const meta = {
    timestamp: new Date().toISOString(),
    caseCount: casesToRun.length,
    tagFilter: tagFilter || null,
  };
  const jsonPath = writeReports(results, meta);
  console.log(chalk.gray(`Report written to ${jsonPath}`));
  console.log(chalk.gray(`Latest snapshot: eval/results/latest.json, eval/results/latest.md\n`));

  const realFails = results.filter(r => !r.pass && !r.knownLimitation);
  if (realFails.length > 0 || results.some(r => r.threw)) {
    process.exitCode = 1; // non-zero exit for CI / pre-push hooks
  }
}

main();
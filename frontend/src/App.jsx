import React, { useState, useRef, useCallback } from "react";
import {
  Search, FileText, Globe, DollarSign, Sparkles, ArrowRight, ArrowRightLeft,
  ShieldCheck, ShieldAlert, AlertTriangle, TimerOff, GitMerge, ChevronDown,
  CheckCircle2, Circle, Github, Brain, Ban,
} from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "https://investment-research-bot-backend.onrender.com";

const INK_950 = "#0A0C10", INK_900 = "#12151C", LINE = "#1E2330";
const PAPER = "#E7E9EF", MUTED = "#6B7280";
const GOLD = "#E8A33D", TEAL = "#4FD1C5", MINT = "#34D399", VIOLET = "#A78BFA", CORAL = "#F2665C";
const FONT_DISPLAY = "'Space Grotesk', sans-serif";
const FONT_DATA = "'IBM Plex Mono', 'JetBrains Mono', monospace";
const FONT_BODY = "'Inter', sans-serif";

const TICKERS = [
  { ticker: "NVDA", name: "NVIDIA" }, { ticker: "AMD", name: "AMD" },
  { ticker: "MSFT", name: "Microsoft" }, { ticker: "AAPL", name: "Apple" },
  { ticker: "GOOGL", name: "Alphabet" }, { ticker: "AMZN", name: "Amazon" },
  { ticker: "TSLA", name: "Tesla" }, { ticker: "META", name: "Meta" },
];

const SAMPLE_QUERIES = [
  "Compare Tesla and AMD's recent financials",
  "Analyze Tesla: recent financials, news sentiment, and stock performance",
  "What was Apple's operating margin last year?",
  "Compare NVIDIA and AMD's data center revenue growth and stock performance",
];

const TOOL_META = {
  RAG: { icon: FileText, color: GOLD, label: "RAG" },
  WEB: { icon: Globe, color: TEAL, label: "WEB" },
  STOCK: { icon: DollarSign, color: MINT, label: "STOCK" },
};

const TOOL_ACTIVITY_LABEL = {
  RAG: "RAG: reading filings…",
  WEB: "WEB: searching the web…",
  STOCK: "STOCK: fetching market data…",
};

const VERDICT_META = {
  sufficient: { label: "VERIFIED", color: MINT, Icon: ShieldCheck },
  needs_more_evidence: { label: "GAP FOUND", color: GOLD, Icon: AlertTriangle },
  unsupported_claim: { label: "FLAGGED", color: CORAL, Icon: ShieldAlert },
  conflicting_evidence: { label: "CONFLICT", color: CORAL, Icon: ShieldAlert },
  cap_reached: { label: "CAPPED", color: MUTED, Icon: TimerOff },
  verification_unavailable: { label: "UNAVAILABLE", color: MUTED, Icon: TimerOff },
};

function timestamp() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

function FilingStamp({ verdict, small }) {
  const meta = VERDICT_META[verdict] || VERDICT_META.sufficient;
  const Icon = meta.Icon;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-sm border-2 px-2 py-0.5 uppercase tracking-widest select-none whitespace-nowrap ${small ? "text-[8.5px]" : "text-[10px]"}`}
      style={{ color: meta.color, borderColor: meta.color, fontFamily: FONT_DATA, transform: "rotate(-3deg)" }}
    >
      <Icon className={small ? "w-2.5 h-2.5" : "w-3 h-3"} /> {meta.label}
    </span>
  );
}

function ToolBadge({ tool }) {
  const meta = TOOL_META[tool] || TOOL_META.RAG;
  const Icon = meta.icon;
  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-[10.5px] tracking-wide uppercase"
      style={{ color: meta.color, borderColor: `${meta.color}4D`, background: `${meta.color}0D`, fontFamily: FONT_DATA }}>
      <Icon className="w-3 h-3" /> {meta.label}
    </span>
  );
}

/* "Agent is still working" indicator — three staggered dots instead of a
   static blinking square, reads as "processing" rather than "typing". */
function WorkingIndicator() {
  return (
    <div className="flex items-center gap-1.5 pl-6 py-1">
      {[0, 1, 2].map(i => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full animate-bounce"
          style={{ background: GOLD, animationDelay: `${i * 0.15}s`, animationDuration: "0.9s" }}
        />
      ))}
    </div>
  );
}

function TraceBody({ entry }) {
  if (entry.node === "router") {
    return (
      <div className="mt-1.5 space-y-2">
        {(entry.requiredTools || []).length > 0 && (
          <div className="flex gap-1.5">{entry.requiredTools.map(t => <ToolBadge key={t} tool={t} />)}</div>
        )}
        {(entry.subQuestions || []).map((sq, i) => (
          <div key={i} className="pl-2.5" style={{ borderLeft: `1px solid ${LINE}` }}>
            <p className="text-[11px]" style={{ color: MUTED, fontFamily: FONT_DATA }}>{sq.question}</p>
            <p className="text-[10.5px] mt-0.5" style={{ color: "#4B5563", fontFamily: FONT_DATA }}>
              → <span style={{ color: GOLD }}>{sq.tool}</span>
              {sq.companies?.length ? ` · ${sq.companies.join(", ")}` : ""}
              {sq.years?.length ? ` · FY${sq.years.join("/")}` : ""}
            </p>
          </div>
        ))}
      </div>
    );
  }

  if (entry.node === "rag" || entry.node === "web" || entry.node === "stock") {
    const meta = TOOL_META[entry.node.toUpperCase()];
    return (
      <div className="mt-1">
        <p className="text-[11px]" style={{ color: MUTED, fontFamily: FONT_DATA }}>{entry.question}</p>
        <p className="text-[10.5px] mt-0.5" style={{ color: meta.color, fontFamily: FONT_DATA }}>
          {entry.node === "rag" && `${entry.xbrlFactCount ?? 0} xbrl-fact + ${entry.narrativeCandidateCount ?? 0} candidates → ${entry.rerankedCount ?? 0} kept`}
          {entry.node === "web" && `${entry.articlesFound ?? 0} article(s) retrieved`}
          {entry.node === "stock" && `${entry.resultCount ?? 0} result(s)`}
          {entry.error && <span style={{ color: CORAL }}> · failed: {entry.error}</span>}
        </p>
      </div>
    );
  }

  if (entry.node === "verifier" || entry.node === "globalCheck") {
    if (entry.skipped) return <p className="mt-1 text-[10.5px]" style={{ color: MUTED, fontFamily: FONT_DATA }}>skipped — {entry.reason}</p>;
    return (
      <div className="mt-1.5">
        <FilingStamp verdict={entry.verdict} small />
        <p className="text-[11px] mt-1.5 leading-relaxed" style={{ color: "#9CA3AF" }}>{entry.reasoning}</p>
      </div>
    );
  }

  if (entry.node === "resynthesize") {
    return <p className="mt-1 text-[11px]" style={{ color: VIOLET, fontFamily: FONT_DATA }}>rewriting to correct: {entry.correctedVerdict}</p>;
  }

  if (entry.node === "capReached") {
    return <p className="mt-1 text-[11px]" style={{ color: MUTED, fontFamily: FONT_DATA }}>retry budget reached after {entry.attempts} attempt(s)</p>;
  }

  if (entry.node === "gapFill:escalated") {
    return (
      <p className="mt-1 text-[10.5px] flex items-center gap-1.5" style={{ fontFamily: FONT_DATA }}>
        <span style={{ color: TOOL_META[entry.from]?.color }}>{entry.from}</span>
        <ArrowRight className="w-2.5 h-2.5" style={{ color: MUTED }} />
        <span style={{ color: TOOL_META[entry.to]?.color }}>{entry.to}</span>
        <span style={{ color: MUTED }}>— {entry.gap}</span>
      </p>
    );
  }

  if (entry.node === "gapFill:abandoned") {
    return <p className="mt-1 text-[10.5px]" style={{ color: MUTED, fontFamily: FONT_DATA }}><Ban className="w-2.5 h-2.5 inline mr-1" />left unresolved: {entry.gap}</p>;
  }

  return null;
}

const NODE_LABEL = {
  router: "Router", rag: "RAG", web: "Web", stock: "Stock",
  verifier: "Verifier", globalCheck: "Reconciliation", resynthesize: "Correcting",
  capReached: "Retry cap", "gapFill:escalated": "Escalated", "gapFill:abandoned": "Abandoned",
  finalizeMultiEntity: "Merging briefs",
};

function TraceRow({ entry }) {
  return (
    <div className="flex items-start gap-2.5">
      <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: MINT }} />
      <div className="min-w-0 flex-1">
        <p style={{ color: MUTED }}>
          <span style={{ color: "#4B5563" }}>[{entry.time}]</span>{" "}
          <span style={{ color: PAPER }}>{NODE_LABEL[entry.node] || entry.node}</span>
        </p>
        <TraceBody entry={entry} />
      </div>
    </div>
  );
}

function CompanyLane({ ticker, lane }) {
  const status = lane?.status || "pending";
  const events = lane?.events || [];
  const lastVerdict = [...events].reverse().find(e => e.kind === "verifier")?.verdict;
  const capped = events.some(e => e.kind === "capReached");

  return (
    <div className="rounded-lg border flex-shrink-0 w-[280px]" style={{ borderColor: LINE, background: INK_900 }}>
      <div className="px-4 py-3 flex items-center justify-between" style={{ borderBottom: `1px solid ${LINE}` }}>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full flex-shrink-0 ${status === "active" ? "animate-pulse" : ""}`}
            style={{ background: status === "done" ? MINT : status === "active" ? GOLD : MUTED }} />
          <span className="text-[13px] font-semibold tracking-wide" style={{ color: PAPER, fontFamily: FONT_DATA }}>{ticker}</span>
        </div>
        {status === "done" && lastVerdict && <FilingStamp verdict={capped ? "cap_reached" : lastVerdict} small />}
      </div>
      <div className="p-3 max-h-[280px] overflow-y-auto">
        {status === "pending" && <p className="text-[11px]" style={{ color: MUTED, fontFamily: FONT_DATA }}>queued…</p>}
        {status === "active" && <WorkingIndicator />}
        {status === "done" && events.length === 0 && (
          <p className="text-[11px]" style={{ color: MUTED, fontFamily: FONT_DATA }}>resolved on first pass — no gaps or corrections needed</p>
        )}
        {events.map((e, i) => (
          <div key={i} className="text-[11px] py-1" style={{ borderBottom: i < events.length - 1 ? `1px solid ${LINE}` : "none" }}>
            {e.kind === "verifier" && <FilingStamp verdict={e.verdict} small />}
            {e.kind === "escalated" && <p style={{ color: MUTED, fontFamily: FONT_DATA }}><span style={{ color: TOOL_META[e.from]?.color }}>{e.from}</span> → <span style={{ color: TOOL_META[e.to]?.color }}>{e.to}</span></p>}
            {e.kind === "abandoned" && <p style={{ color: MUTED, fontFamily: FONT_DATA }}>gap unresolved</p>}
            {e.kind === "capReached" && <p style={{ color: MUTED, fontFamily: FONT_DATA }}>retry budget reached ({e.attempts} attempts)</p>}
          </div>
        ))}
      </div>
    </div>
  );
}

function EvidenceSource({ source }) {
  if (source.url) {
    return (
      <a href={source.url} target="_blank" rel="noopener noreferrer" title={source.title}
        className="px-1.5 py-0.5 border rounded hover:opacity-80 transition-opacity truncate max-w-[240px] inline-block cursor-pointer"
        style={{ borderColor: LINE, color: TEAL, fontFamily: FONT_DATA }}>
        {source.title || source.url}
      </a>
    );
  }
  return <span className="px-1.5 py-0.5 border rounded" style={{ borderColor: LINE, color: MUTED, fontFamily: FONT_DATA }}>
    {source.company || source.ticker} {source.year || ""}{source.section ? ` · ${source.section}` : ""}
  </span>;
}

function ActivityLine({ label }) {
  if (!label) return null;
  return (
    <p className="text-[11px] uppercase tracking-wide animate-pulse mt-3 flex items-center gap-1.5" style={{ color: VIOLET, fontFamily: FONT_DATA }}>
      <Brain className="w-3 h-3" /> {label}
    </p>
  );
}

export default function InvestmentResearchUI() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [activityLabel, setActivityLabel] = useState(null);

  const [originalQuestion, setOriginalQuestion] = useState("");
  const [requiredTools, setRequiredTools] = useState([]);
  const [lanes, setLanes] = useState({});
  const [trace, setTrace] = useState([]);
  const [evidence, setEvidence] = useState([]);
  const [answer, setAnswer] = useState(null);
  const [revision, setRevision] = useState(0);
  const [finalVerdict, setFinalVerdict] = useState(null);

  const idCounter = useRef(0);
  const resultsRef = useRef(null);
  const isMultiEntity = Object.keys(lanes).length > 1;

  const reset = () => {
    setError(null); setRequiredTools([]); setLanes({}); setTrace([]);
    setEvidence([]); setAnswer(null); setRevision(0); setFinalVerdict(null);
    idCounter.current = 0;
  };

  const pushTrace = useCallback((entry) => {
    setTrace(prev => [...prev, { id: idCounter.current++, time: timestamp(), ...entry }]);
  }, []);

  const mergeEvidence = useCallback((delta) => {
    if (!Array.isArray(delta) || delta.length === 0) return;
    setEvidence(prev => [...prev, ...delta]);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim() || status === "running") return;

    reset();
    setOriginalQuestion(query);
    setStatus("running");
    setActivityLabel("Router model is thinking…");
    setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth" }), 50);

    const url = `${API_BASE}/api/research/stream?question=${encodeURIComponent(query)}`;
    const es = new EventSource(url);

    // Tracks which of the router's requiredTools have actually completed,
    // in ORDER — this is what makes the activity line say "WEB: searching…"
    // instead of collapsing every tool step into "Synthesizing answer…".
    // Local to this run, not state — no need to re-render on every mutation,
    // only the derived label matters.
    let toolsDone = [];
    let toolPipeline = [];

    es.addEventListener("step", (e) => {
      const { node, company, payload } = JSON.parse(e.data);

      if (node === "classify") {
        setRequiredTools(payload.requiredTools || []);
        toolPipeline = payload.requiredTools || [];
        pushTrace({ node: "router", requiredTools: payload.requiredTools, subQuestions: payload.subQuestions });

        const uniq = [...new Set((payload.subQuestions || []).flatMap(q => q.companies || []))];
        if (uniq.length > 1) {
          setLanes(Object.fromEntries(uniq.map(c => [c, { status: "active", events: [] }])));
          setActivityLabel("Per-company sub-agents retrieving & self-verifying…");
        } else {
          setActivityLabel(TOOL_ACTIVITY_LABEL[toolPipeline[0]] || "Synthesizing answer…");
        }
        return;
      }

      if (node === "subAgent" && company) {
        const companyEvents = (payload.agentTrace || [])
          .filter(entry => entry.node?.startsWith(`subAgent:${company}:`))
          .map(entry => ({ ...entry, kind: entry.node.split(":")[2] }));

        let stillRunning = false;
        setLanes(prev => {
          const next = { ...prev, [company]: { status: "done", events: companyEvents } };
          stillRunning = Object.values(next).some(l => l.status !== "done");
          return next;
        });
        mergeEvidence(payload.evidence);
        setActivityLabel(stillRunning ? "Per-company sub-agents retrieving & self-verifying…" : "Reconciling briefs across companies…");
        return;
      }

      if (node === "rag" || node === "web" || node === "stock") {
        (payload.agentTrace || []).forEach(entry => pushTrace(entry));
        mergeEvidence(payload.evidence);

        const justFinished = node.toUpperCase();
        if (!toolsDone.includes(justFinished)) toolsDone.push(justFinished);
        const nextTool = toolPipeline.find(t => !toolsDone.includes(t));
        setActivityLabel(nextTool ? TOOL_ACTIVITY_LABEL[nextTool] : "Synthesizing answer…");
        return;
      }

      if (node === "synthesizer" || node === "finalizeMultiEntity") {
        // Both feed straight into the verifier next — this is the
        // dedicated pre-verifier state that was missing before.
        setActivityLabel("Checking grounding…");
      }

      if (node === "verifier" || node === "globalCheck") {
        if (payload?.verifierVerdict) {
          pushTrace({ node, ...payload.verifierVerdict });
          setFinalVerdict(payload.verifierVerdict);
          const v = payload.verifierVerdict.verdict;
          if (node === "verifier") {
            setActivityLabel(
              v === "sufficient" ? null
              : v === "needs_more_evidence" ? "Filling evidence gaps…"
              : "Rewriting to correct an issue…"
            );
          } else {
            setActivityLabel("Merging company briefs…"); // globalCheck always -> finalize next
          }
        } else {
          const skip = payload.agentTrace?.[0];
          if (skip?.skipped) pushTrace({ node, skipped: true, reason: skip.reason });
          setActivityLabel("Merging company briefs…");
        }
        mergeEvidence(payload.evidence);
        return;
      }

      if (node === "gapFill") {
        (payload.agentTrace || []).forEach(entry => pushTrace(entry));
        mergeEvidence(payload.evidence);
        setActivityLabel("Re-synthesizing with new evidence…");
        return;
      }

      if (node === "resynthesize") {
        (payload.agentTrace || []).forEach(entry => pushTrace(entry));
        mergeEvidence(payload.evidence);
        setActivityLabel("Checking grounding…"); // resynthesize -> verifier next
        return;
      }

      if (node === "capReached") {
        (payload.agentTrace || []).forEach(entry => pushTrace(entry));
        mergeEvidence(payload.evidence);
        setActivityLabel(null);
        return;
      }

      mergeEvidence(payload.evidence);
      if (payload.finalOutput) { setAnswer(payload.finalOutput); setRevision(r => r + 1); }
    });

    es.addEventListener("error", (e) => {
      let message = "Connection to research agent failed";
      try { message = JSON.parse(e.data)?.message || message; } catch {}
      setError(message); setStatus("error"); setActivityLabel(null); es.close();
    });

    es.addEventListener("done", () => { setStatus("done"); setActivityLabel(null); es.close(); });
  };

  const markdownContent = typeof answer === "string" ? answer.replace(/^`+|`+$/g, "").trim() : "";
  const hasResults = status !== "idle";

  return (
    <div className="min-h-screen" style={{ background: INK_950, color: PAPER, fontFamily: FONT_BODY }}>
      <header className="border-b sticky top-0 z-50 backdrop-blur-md" style={{ borderColor: LINE, background: `${INK_950}E6` }}>
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <button onClick={() => window.location.reload()} className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity" title="Back to home">
            <div className="w-8 h-8 border rounded flex items-center justify-center gap-[2px]" style={{ borderColor: `${GOLD}66` }}>
              <span className="w-[3px] h-2.5 rounded-sm" style={{ background: `${GOLD}80` }} />
              <span className="w-[3px] h-4 rounded-sm" style={{ background: GOLD }} />
              <span className="w-[3px] h-3 rounded-sm" style={{ background: `${GOLD}B3` }} />
            </div>
            <div className="text-left">
              <h1 className="text-lg font-semibold tracking-tight" style={{ fontFamily: FONT_DISPLAY }}>Investment Research AI</h1>
              <p className="text-[11px]" style={{ color: MUTED, fontFamily: FONT_DATA }}>Self-Fine-Tuned Router · Verified Multi-Agent RAG</p>
            </div>
          </button>
          <div className="flex items-center gap-3">
            <div className="relative group hidden md:block">
              <div className="flex items-center gap-2 px-3 py-1.5 border rounded text-[11px] cursor-default" style={{ borderColor: LINE, color: MUTED, fontFamily: FONT_DATA }}>
                <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: MINT }} /> 8 companies · 15,128 chunks indexed
              </div>
              <div className="absolute right-0 top-full mt-2 w-60 border rounded-lg p-3 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all shadow-xl z-50" style={{ borderColor: LINE, background: INK_900 }}>
                <p className="text-[10px] uppercase tracking-wide mb-2" style={{ color: MUTED, fontFamily: FONT_DATA }}>Companies covered</p>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
                  {TICKERS.map(c => <div key={c.ticker} className="text-[11px]" style={{ fontFamily: FONT_DATA }}><span style={{ color: GOLD }}>{c.ticker}</span> <span style={{ color: MUTED }}>{c.name}</span></div>)}
                </div>
              </div>
            </div>
            <a href="https://huggingface.co/IbrahimKhan7208/investment-research-router" target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 border rounded text-[11px] hover:opacity-80 hover:border-amber-400/40 transition-all cursor-pointer" style={{ borderColor: LINE, color: PAPER, fontFamily: FONT_DATA }}>
              <Brain className="w-3.5 h-3.5" /> Router Model
            </a>
            <a href="https://github.com/IbrahimKhan7208/investment-research-bot" target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 border rounded text-[11px] hover:opacity-80 hover:border-amber-400/40 transition-all cursor-pointer" style={{ borderColor: LINE, color: PAPER, fontFamily: FONT_DATA }}>
              <Github className="w-3.5 h-3.5" /> GitHub
            </a>
          </div>
        </div>
      </header>

      <section className="min-h-[calc(100vh-73px)] flex flex-col justify-center px-6">
        <div className="max-w-3xl mx-auto text-center w-full">
          <p className="text-[11px] tracking-[0.2em] uppercase mb-4" style={{ color: GOLD, fontFamily: FONT_DATA }}>
            SEC Filings · Market Data · Self-Verified Analysis
          </p>
          <h2 className="text-4xl md:text-5xl font-semibold mb-5 leading-[1.1] tracking-tight" style={{ fontFamily: FONT_DISPLAY }}>
            Ask the question.<br />Watch it check its own work.
          </h2>
          <p className="max-w-xl mx-auto mb-10" style={{ color: MUTED }}>
            A router model dispatches your question to SEC filings, live market data, or the web —
            then a second model verifies every claim before it reaches you, and corrects what it can't support.
          </p>
          <form onSubmit={handleSubmit} className="relative mb-6">
            <Search className="absolute left-5 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: MUTED }} />
            <input
              type="text" value={query} onChange={(e) => setQuery(e.target.value)}
              placeholder="Compare Tesla and AMD's recent financials..."
              disabled={status === "running"}
              className="w-full pl-12 pr-32 py-4 rounded-lg text-[15px] focus:outline-none transition-colors disabled:opacity-50"
              style={{ background: INK_900, border: `1px solid ${LINE}`, color: PAPER }}
            />
            <button type="submit" disabled={!query.trim() || status === "running"}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-2 rounded-md font-medium text-sm transition-opacity disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer hover:opacity-90"
              style={{ background: GOLD, color: INK_950 }}>
              {status === "running" ? "Analyzing…" : "Analyze"}
            </button>
          </form>
          <div className="flex flex-wrap gap-2 justify-center">
            {SAMPLE_QUERIES.map((s, i) => (
              <button key={i} onClick={() => setQuery(s)} disabled={status === "running"}
                className="text-xs px-3 py-1.5 border rounded-full transition-colors cursor-pointer hover:border-amber-400/40 hover:text-amber-400 disabled:opacity-40 disabled:cursor-not-allowed"
                style={{ borderColor: LINE, color: MUTED, fontFamily: FONT_DATA }}>
                {s}
              </button>
            ))}
          </div>
          {hasResults && (
            <button onClick={() => resultsRef.current?.scrollIntoView({ behavior: "smooth" })}
              className="mt-12 text-[11px] font-mono animate-bounce cursor-pointer hover:opacity-80 transition-opacity"
              style={{ color: MUTED, fontFamily: FONT_DATA }}>
              ↓ view analysis
            </button>
          )}
        </div>
      </section>

      {error && (
        <div className="max-w-3xl mx-auto px-6 mb-10 border rounded-lg p-4 flex items-start gap-3" style={{ borderColor: `${CORAL}4D`, background: `${CORAL}0D` }}>
          <ShieldAlert className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: CORAL }} />
          <p className="text-sm" style={{ color: CORAL }}>{error}</p>
        </div>
      )}

      {hasResults && (
        <section ref={resultsRef} className="border-t px-6 py-16" style={{ borderColor: LINE, background: "#0A0D13" }}>
          <div className="max-w-7xl mx-auto space-y-6">

            {isMultiEntity && (
              <div>
                <p className="text-[10px] uppercase tracking-wide mb-3" style={{ color: MUTED, fontFamily: FONT_DATA }}>Per-company sub-agents (parallel)</p>
                <div className="flex gap-4 overflow-x-auto pb-2">
                  {Object.entries(lanes).map(([ticker, lane]) => <CompanyLane key={ticker} ticker={ticker} lane={lane} />)}
                </div>
                <div className="flex items-center gap-2 mt-4 mb-1" style={{ color: MUTED }}>
                  <GitMerge className="w-3.5 h-3.5" />
                  <span className="text-[10px] uppercase tracking-wide" style={{ fontFamily: FONT_DATA }}>converges into reconciliation & final verification below</span>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6 items-start">
              <div className="rounded-lg sticky top-24" style={{ background: "#0D1017", border: `1px solid ${LINE}` }}>
                <div className="px-4 py-3 flex items-center gap-2" style={{ borderBottom: `1px solid ${LINE}` }}>
                  <div className="flex gap-1.5">
                    <span className="w-2 h-2 rounded-full" style={{ background: "#EF444499" }} />
                    <span className="w-2 h-2 rounded-full" style={{ background: `${GOLD}99` }} />
                    <span className="w-2 h-2 rounded-full" style={{ background: `${MINT}99` }} />
                  </div>
                  <p className="text-[11px] uppercase tracking-wide ml-1" style={{ color: MUTED, fontFamily: FONT_DATA }}>agent trace</p>
                </div>
                <div className="p-4 text-[12px] space-y-4 max-h-[600px] overflow-y-auto" style={{ fontFamily: FONT_DATA }}>
                  {trace.map(entry => <TraceRow key={entry.id} entry={entry} />)}
                  {status === "running" && <WorkingIndicator />}
                </div>
              </div>

              <div className="space-y-6">
                <div className="rounded-lg p-5" style={{ background: INK_900, border: `1px solid ${LINE}` }}>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-[10px] uppercase tracking-wide mb-2" style={{ color: MUTED, fontFamily: FONT_DATA }}>Question</p>
                      <p className="text-[15px]">{originalQuestion}</p>
                      {requiredTools.length > 0 && <div className="flex gap-2 mt-3">{requiredTools.map(t => <ToolBadge key={t} tool={t} />)}</div>}
                    </div>
                    {status === "done" && <FilingStamp verdict={finalVerdict?.verdict || "sufficient"} />}
                  </div>
                  {status === "running" && <ActivityLine label={activityLabel} />}
                </div>

                {answer && (
                  <div className="rounded-lg p-6" style={{ background: INK_900, border: `1px solid ${LINE}` }}>
                    <div className="flex items-center justify-between mb-4">
                      <p className="text-[11px] uppercase tracking-wide" style={{ color: MINT, fontFamily: FONT_DATA }}>Answer</p>
                      <div className="flex items-center gap-2">
                        {revision > 1 && <span className="text-[10px] uppercase tracking-wide" style={{ color: MUTED, fontFamily: FONT_DATA }}>revision {revision}</span>}
                        {status === "done" && <FilingStamp verdict={finalVerdict?.verdict || "sufficient"} small />}
                      </div>
                    </div>
                    <div className="prose-invert max-w-none">
                      <Markdown remarkPlugins={[remarkGfm]} components={{
                        table: (p) => <div className="overflow-x-auto my-4 rounded-md" style={{ border: `1px solid ${LINE}` }}><table className="w-full border-collapse text-sm" {...p} /></div>,
                        thead: (p) => <thead style={{ background: "#1A1F2B" }} {...p} />,
                        th: (p) => <th className="px-3 py-2 text-left text-[11px] uppercase tracking-wide" style={{ border: `1px solid ${LINE}`, color: GOLD, fontFamily: FONT_DATA }} {...p} />,
                        td: (p) => <td className="px-3 py-2 align-top" style={{ border: `1px solid ${LINE}`, fontFamily: FONT_DATA }} {...p} />,
                        p: (p) => <p className="mb-3 leading-relaxed" {...p} />,
                        strong: (p) => <strong className="font-semibold" style={{ color: PAPER }} {...p} />,
                        ul: (p) => <ul className="list-disc pl-5 mb-3 space-y-1.5" {...p} />,
                        ol: (p) => <ol className="list-decimal pl-5 mb-3 space-y-1.5" {...p} />,
                        h1: (p) => <h3 className="text-base font-semibold mt-5 mb-2" style={{ fontFamily: FONT_DISPLAY }} {...p} />,
                        h2: (p) => <h3 className="text-base font-semibold mt-5 mb-2" style={{ fontFamily: FONT_DISPLAY }} {...p} />,
                        h3: (p) => <h4 className="text-sm font-semibold mt-4 mb-1.5" style={{ color: GOLD }} {...p} />,
                      }}>{markdownContent}</Markdown>
                    </div>
                  </div>
                )}

                {evidence.length > 0 && (
                  <div className="rounded-lg" style={{ background: INK_900, border: `1px solid ${LINE}` }}>
                    <details>
                      <summary className="cursor-pointer px-6 py-4 flex items-center gap-2 text-[11px] uppercase tracking-wide hover:opacity-80 transition-opacity" style={{ color: MUTED, fontFamily: FONT_DATA }}>
                        Evidence ({evidence.length}) <span className="ml-auto text-[10px] normal-case">click to expand</span>
                      </summary>
                      <div className="px-6 pb-6 space-y-3">
                        {evidence.map((ev, i) => (
                          <div key={i} className="rounded-md p-4" style={{ background: INK_950, border: `1px solid ${LINE}` }}>
                            <div className="flex items-start gap-2 mb-2"><ToolBadge tool={ev.tool} /><p className="text-xs flex-1" style={{ color: MUTED }}>{ev.question}</p></div>
                            <p className="text-sm mb-2 whitespace-pre-wrap" style={{ color: PAPER }}>{ev.answer}</p>
                            {ev.sources?.length > 0 && <div className="flex flex-wrap gap-1.5 text-[10px]">{ev.sources.map((s, si) => <EvidenceSource key={si} source={s} />)}</div>}
                          </div>
                        ))}
                      </div>
                    </details>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      <footer className="border-t" style={{ borderColor: LINE }}>
        <div className="max-w-7xl mx-auto px-6 py-6 text-center text-[11px]" style={{ color: MUTED, fontFamily: FONT_DATA }}>
          PyTorch · Unsloth · CPT · SFT · GRPO · LangGraph · Cohere · Pinecone · Groq · SEC EDGAR · Yahoo Finance
        </div>
      </footer>
    </div>
  );
}
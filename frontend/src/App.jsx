import React, { useState, useRef, useCallback } from "react";
import {
  Search, FileText, Globe, DollarSign, Sparkles,
  ArrowRight, CheckCircle2, AlertCircle, Circle,
  Github, Brain,
} from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "https://investment-research-bot-backend.onrender.com";

const TICKERS = [
  { ticker: "NVDA", name: "NVIDIA" },
  { ticker: "AMD", name: "AMD" },
  { ticker: "MSFT", name: "Microsoft" },
  { ticker: "AAPL", name: "Apple" },
  { ticker: "GOOGL", name: "Alphabet" },
  { ticker: "AMZN", name: "Amazon" },
  { ticker: "TSLA", name: "Tesla" },
  { ticker: "META", name: "Meta" },
];

const SAMPLE_QUERIES = [
  "Compare NVIDIA and AMD's data center revenue growth and recent stock performance",
  "Analyze Tesla: recent financials, news sentiment, and stock performance",
  "What was Apple's operating margin last year?",
  "What are Meta's main competitive risks according to its latest filing?",
];

const NODE_META = {
  classify: { label: "Router", icon: Sparkles },
  rag: { label: "RAG", icon: FileText },
  web: { label: "Web", icon: Globe },
  stock: { label: "Stock", icon: DollarSign },
  synthesizer: { label: "Synthesizer", icon: ArrowRight },
};

const TOOL_META = {
  RAG: { icon: FileText, color: "text-amber-400 border-amber-400/30 bg-amber-400/5" },
  WEB: { icon: Globe, color: "text-teal-400 border-teal-400/30 bg-teal-400/5" },
  STOCK: { icon: DollarSign, color: "text-emerald-400 border-emerald-400/30 bg-emerald-400/5" },
};

function ToolBadge({ tool }) {
  const meta = TOOL_META[tool] || TOOL_META.RAG;
  const Icon = meta.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-[11px] font-mono tracking-wide uppercase ${meta.color}`}>
      <Icon className="w-3 h-3" />
      {tool}
    </span>
  );
}

function timestamp() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

// Renders whatever internal detail a step reported — shape differs per node,
// since the router, RAG, web, and stock nodes each surface different things.
function TraceDetail({ node, detail }) {
  if (!detail) return null;

  if (node === "classify" && detail.subQuestions) {
    return (
      <div className="mt-2 space-y-2">
        {detail.subQuestions.map((sq, i) => (
          <div key={i} className="pl-2.5 border-l border-[#242938]">
            <p className="text-[11px] text-[#9CA3AF]">{sq.question}</p>
            <p className="text-[11px] text-[#4B5563] mt-0.5">
              → <span className="text-amber-400">{sq.tool}</span>
              {sq.companies?.length ? ` · ${sq.companies.join(", ")}` : ""}
              {sq.years?.length ? ` · FY${sq.years.join("/")}` : ""}
              {sq.searchQuery ? ` · "${sq.searchQuery}"` : ""}
            </p>
          </div>
        ))}
      </div>
    );
  }

  if (node === "rag" && detail.trace) {
    return (
      <div className="mt-2 space-y-1.5">
        {detail.trace.map((t, i) => (
          <p key={i} className="text-[11px] text-[#4B5563] pl-2.5 border-l border-[#242938]">
            {t.xbrlFactCount} xbrl-fact <span className="text-[#374151]">(guaranteed)</span> + {t.narrativeCandidateCount} candidates → {t.rerankedCount} kept via rerank
          </p>
        ))}
      </div>
    );
  }

  if (node === "web" && detail.trace) {
    return (
      <div className="mt-2 space-y-1.5">
        {detail.trace.map((t, i) => (
          <p key={i} className="text-[11px] text-[#4B5563] pl-2.5 border-l border-[#242938]">
            Query: <span className="text-[#9CA3AF]">"{t.query}"</span> → {t.articlesFound} article(s) found
          </p>
        ))}
      </div>
    );
  }

  if (node === "stock" && detail.trace) {
    return (
      <div className="mt-2 space-y-1.5">
        {detail.trace.map((t, i) => (
          <p key={i} className="text-[11px] text-[#4B5563] pl-2.5 border-l border-[#242938]">
            Resolved <span className="text-[#9CA3AF]">{t.tickers?.join(", ") || "none"}</span> · type: {t.queryType}
            {t.period ? ` · period: ${t.period}` : ""} · {t.resultCount} result(s)
          </p>
        ))}
      </div>
    );
  }

  if (detail.note) {
    return <p className="text-[11px] text-[#4B5563] pl-2.5 border-l border-[#242938] mt-1">{detail.note}</p>;
  }

  return null;
}

export default function InvestmentResearchUI() {
  const [query, setQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [steps, setSteps] = useState([]);
  const [requiredTools, setRequiredTools] = useState([]);
  const [evidence, setEvidence] = useState([]);
  const [finalOutput, setFinalOutput] = useState(null);
  const [originalQuestion, setOriginalQuestion] = useState("");
  const resultsRef = useRef(null);

  const reset = () => {
    setError(null);
    setSteps([]);
    setRequiredTools([]);
    setEvidence([]);
    setFinalOutput(null);
  };

  const pushStep = useCallback((node, status, detail) => {
    setSteps((prev) => {
      const idx = prev.findIndex((s) => s.node === node);
      const entry = { node, status, detail, time: timestamp() };
      if (idx === -1) return [...prev, entry];
      const copy = [...prev];
      copy[idx] = entry;
      return copy;
    });
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim() || isProcessing) return;

    reset();
    setOriginalQuestion(query);
    setIsProcessing(true);
    pushStep("classify", "active", null);

    const url = `${API_BASE}/api/research/stream?question=${encodeURIComponent(query)}`;
    const es = new EventSource(url);

    es.addEventListener("step", (e) => {
      const { node, payload } = JSON.parse(e.data);

      if (node === "classify") {
        setRequiredTools(payload.requiredTools || []);
        pushStep("classify", "done", { subQuestions: payload.subQuestions });
        (payload.requiredTools || []).forEach((tool) => pushStep(tool.toLowerCase(), "active", null));
      } else if (node === "synthesizer") {
        pushStep("synthesizer", "done", { note: "Answer composed from all retrieved evidence" });
        setFinalOutput(payload.finalOutput);
      } else {
        setEvidence(payload.evidence || []);
        const myTrace = (payload.agentTrace || []).filter((t) => t.node === node);
        pushStep(node, "done", { trace: myTrace });
        pushStep("synthesizer", "active", null);
      }
    });

    es.addEventListener("error", (e) => {
      let message = "Connection to research agent failed";
      try { message = JSON.parse(e.data)?.message || message; } catch {}
      setError(message);
      setIsProcessing(false);
      es.close();
    });

    es.addEventListener("done", () => {
      setIsProcessing(false);
      es.close();
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    });
  };

  const markdownContent = typeof finalOutput === "string" ? finalOutput.replace(/^`+|`+$/g, "").trim() : "";
  const hasResults = steps.length > 0 || finalOutput;

  return (
    <div className="min-h-screen bg-[#0B0E14] text-[#E4E7EC]" style={{ fontFamily: "'Inter', sans-serif" }}>
      {/* Header */}
      <header className="border-b border-[#242938] bg-[#0B0E14]/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 border border-amber-400/40 rounded flex items-center justify-center">
              <div className="w-8 h-8 border border-amber-400/40 rounded flex items-center justify-center gap-[2px]">
                <span className="w-[3px] h-2.5 bg-amber-400/50 rounded-sm" />
                <span className="w-[3px] h-4 bg-amber-400 rounded-sm" />
                <span className="w-[3px] h-3 bg-amber-400/70 rounded-sm" />
              </div>
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                Investment Research AI
              </h1>
              <p className="text-[11px] text-[#6B7280] font-mono">Self-Fine-Tuned Router · Multi-Agent RAG</p>
            </div>
          </div>

          {/* Hover reveals coverage — no more permanent grid taking up space */}
          <div className="flex items-center gap-3">
            <div className="relative group hidden md:block">
              <div className="flex items-center gap-2 px-3 py-1.5 border border-[#242938] rounded font-mono text-[11px] text-[#6B7280] cursor-default hover:border-amber-400/30 transition-colors">
                <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                8 companies · 15,128 chunks indexed
              </div>
              <div className="absolute right-0 top-full mt-2 w-60 border border-[#242938] rounded-lg bg-[#12161F] p-3 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all shadow-xl z-50">
                <p className="text-[10px] text-[#6B7280] uppercase tracking-wide mb-2 font-mono">Companies covered</p>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
                  {TICKERS.map((c) => (
                    <div key={c.ticker} className="text-[11px]">
                      <span className="text-amber-400 font-mono">{c.ticker}</span>
                      <span className="text-[#6B7280]"> · {c.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            
              <a href="https://huggingface.co/IbrahimKhan7208/investment-research-router"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 border border-[#242938] rounded font-mono text-[11px] text-[#9CA3AF] hover:border-amber-400/40 hover:text-amber-400 transition-colors"
            >
              <Brain className="w-3.5 h-3.5" />
              Router Model
            </a>

            
              <a href="https://github.com/IbrahimKhan7208/investment-research-bot"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 border border-[#242938] rounded font-mono text-[11px] text-[#9CA3AF] hover:border-amber-400/40 hover:text-amber-400 transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Hero — fills the initial viewport, no results competing for space */}
      <section className="min-h-[calc(100vh-73px)] flex flex-col justify-center px-6">
        <div className="max-w-3xl mx-auto text-center w-full">
          <p className="font-mono text-[11px] tracking-[0.2em] text-amber-400 uppercase mb-4">
            SEC Filings · Market Data · Live Analysis
          </p>
          <h2
            className="text-4xl md:text-5xl font-semibold mb-5 leading-[1.1] tracking-tight"
            style={{ fontFamily: "'Space Grotesk', sans-serif" }}
          >
            Ask the question.<br />Watch it get answered.
          </h2>
          <p className="text-[#9CA3AF] max-w-xl mx-auto mb-10">
            My self-fine-tuned router model (Qwen2.5-1.5B, trained CPT → SFT → GRPO) directs your question to
            SEC filings, live market data, or the web — then shows its work at every step.
          </p>

          <form onSubmit={handleSubmit} className="relative mb-6">
            <Search className="absolute left-5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#6B7280]" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Compare NVIDIA and AMD data center revenue growth..."
              disabled={isProcessing}
              className="w-full pl-12 pr-32 py-4 bg-[#12161F] border border-[#242938] rounded-lg text-[15px] focus:outline-none focus:border-amber-400/50 transition-colors placeholder:text-[#4B5563] disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!query.trim() || isProcessing}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-2 bg-amber-400 text-[#0B0E14] rounded-md font-medium text-sm hover:bg-amber-300 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isProcessing ? "Analyzing…" : "Analyze"}
            </button>
          </form>

          <div className="flex flex-wrap gap-2 justify-center">
            {SAMPLE_QUERIES.map((s, i) => (
              <button
                key={i}
                onClick={() => setQuery(s)}
                disabled={isProcessing}
                className="text-xs px-3 py-1.5 border border-[#242938] rounded-full text-[#9CA3AF] hover:border-amber-400/40 hover:text-amber-400 transition-colors disabled:opacity-40"
              >
                {s}
              </button>
            ))}
          </div>

          {hasResults && (
            <button
              onClick={() => resultsRef.current?.scrollIntoView({ behavior: "smooth" })}
              className="mt-12 text-[11px] font-mono text-[#4B5563] hover:text-amber-400 transition-colors animate-bounce"
            >
              ↓ view analysis
            </button>
          )}
        </div>
      </section>

      {/* Error */}
      {error && (
        <div className="max-w-3xl mx-auto px-6 mb-10 border border-red-500/30 bg-red-500/5 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Results — a distinct section below the fold */}
      {hasResults && (
        <section ref={resultsRef} className="border-t border-[#242938] bg-[#0A0D13] px-6 py-16">
          <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6 items-start">
            {/* Chain of thought — live terminal trace */}
            <div className="border border-[#242938] rounded-lg bg-[#0D1017] sticky top-24">
              <div className="px-4 py-3 border-b border-[#242938] flex items-center gap-2">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-red-500/60" />
                  <span className="w-2 h-2 rounded-full bg-amber-400/60" />
                  <span className="w-2 h-2 rounded-full bg-emerald-400/60" />
                </div>
                <p className="font-mono text-[11px] text-[#6B7280] tracking-wide uppercase ml-1">agent trace</p>
              </div>
              <div className="p-4 font-mono text-[12px] space-y-4 max-h-[600px] overflow-y-auto">
                {steps.map((s, i) => {
                  const meta = NODE_META[s.node];
                  const Icon = meta?.icon || Circle;
                  return (
                    <div key={i} className="flex items-start gap-2.5">
                      {s.status === "done" ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                      ) : (
                        <Icon className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0 animate-pulse" />
                      )}
                      <div className="min-w-0 flex-1">
                        <p className="text-[#6B7280]">
                          <span className="text-[#4B5563]">[{s.time}]</span>{" "}
                          <span className={s.status === "done" ? "text-[#D1D5DB]" : "text-amber-400"}>
                            {meta?.label || s.node}
                          </span>
                        </p>
                        <TraceDetail node={s.node} detail={s.detail} />
                      </div>
                    </div>
                  );
                })}
                {isProcessing && (
                  <div className="flex items-center gap-1 text-amber-400 pl-6">
                    <span className="w-1.5 h-3 bg-amber-400 animate-pulse" />
                  </div>
                )}
              </div>
            </div>

            {/* Answer + evidence */}
            <div className="space-y-6">
              {originalQuestion && (
                <div className="border border-[#242938] rounded-lg p-5 bg-[#12161F]">
                  <p className="font-mono text-[11px] text-[#6B7280] uppercase tracking-wide mb-2">Question</p>
                  <p className="text-[15px] text-[#E4E7EC]">{originalQuestion}</p>
                  {requiredTools.length > 0 && (
                    <div className="flex gap-2 mt-3">
                      {requiredTools.map((t) => <ToolBadge key={t} tool={t} />)}
                    </div>
                  )}
                </div>
              )}

              {finalOutput && (
                <div className="border border-[#242938] rounded-lg p-6 bg-[#12161F]">
                  <p className="font-mono text-[11px] text-emerald-400 uppercase tracking-wide mb-4">Answer</p>
                  <div className="prose-invert max-w-none">
                    <Markdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        table: ({ node, ...props }) => (
                          <div className="overflow-x-auto my-4 rounded-md border border-[#242938]">
                            <table className="w-full border-collapse text-sm" {...props} />
                          </div>
                        ),
                        thead: ({ node, ...props }) => <thead className="bg-[#1A1F2B]" {...props} />,
                        th: ({ node, ...props }) => (
                          <th className="border border-[#242938] px-3 py-2 text-left font-mono text-[11px] text-amber-400 uppercase tracking-wide" {...props} />
                        ),
                        td: ({ node, ...props }) => (
                          <td className="border border-[#242938] px-3 py-2 text-[#D1D5DB] align-top" {...props} />
                        ),
                        tr: ({ node, ...props }) => <tr className="even:bg-[#0D1017]" {...props} />,
                        p: ({ node, ...props }) => <p className="mb-3 leading-relaxed text-[#D1D5DB]" {...props} />,
                        strong: ({ node, ...props }) => <strong className="text-[#E4E7EC] font-semibold" {...props} />,
                        ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-3 space-y-1.5 text-[#D1D5DB]" {...props} />,
                        ol: ({ node, ...props }) => <ol className="list-decimal pl-5 mb-3 space-y-1.5 text-[#D1D5DB]" {...props} />,
                        li: ({ node, ...props }) => <li className="leading-relaxed" {...props} />,
                        h1: ({ node, ...props }) => <h3 className="text-base font-semibold mt-5 mb-2 text-[#E4E7EC]" style={{ fontFamily: "'Space Grotesk', sans-serif" }} {...props} />,
                        h2: ({ node, ...props }) => <h3 className="text-base font-semibold mt-5 mb-2 text-[#E4E7EC]" style={{ fontFamily: "'Space Grotesk', sans-serif" }} {...props} />,
                        h3: ({ node, ...props }) => <h4 className="text-sm font-semibold mt-4 mb-1.5 text-amber-400" {...props} />,
                      }}
                    >
                      {markdownContent}
                    </Markdown>
                  </div>
                </div>
              )}

              {evidence.length > 0 && (
                <div className="border border-[#242938] rounded-lg bg-[#12161F]">
                  <details className="group">
                    <summary className="cursor-pointer px-6 py-4 flex items-center gap-2 font-mono text-[11px] text-[#6B7280] uppercase tracking-wide">
                      Evidence ({evidence.length})
                      <span className="ml-auto text-[10px] normal-case text-[#4B5563]">click to expand</span>
                    </summary>
                    <div className="px-6 pb-6 space-y-3">
                      {evidence.map((ev, i) => (
                        <div key={i} className="border border-[#242938] rounded-md p-4 bg-[#0D1017]">
                          <div className="flex items-start gap-2 mb-2">
                            <ToolBadge tool={ev.tool} />
                            <p className="text-xs text-[#6B7280] flex-1">{ev.question}</p>
                          </div>
                          <p className="text-sm text-[#D1D5DB] mb-2 whitespace-pre-wrap">{ev.answer}</p>
                          {ev.sources?.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 font-mono text-[10px] text-[#6B7280]">
                              {ev.sources.map((s, si) => (
                                <span key={si} className="px-1.5 py-0.5 border border-[#242938] rounded">
                                  {s.company || s.ticker} {s.year || ""} {s.section ? `· ${s.section}` : ""} {s.sourceType ? `· ${s.sourceType}` : ""}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </details>
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      <footer className="border-t border-[#242938]">
        <div className="max-w-7xl mx-auto px-6 py-6 text-center text-[11px] font-mono text-[#4B5563]">
          PyTorch · Unsloth · CPT · SFT · GRPO · LangGraph · Cohere · Pinecone · Groq · SEC EDGAR · Yahoo Finance
        </div>
      </footer>
    </div>
  );
}
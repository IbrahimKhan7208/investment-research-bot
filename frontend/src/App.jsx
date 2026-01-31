import React, { useState, useRef } from "react";
import {
  Search,
  TrendingUp,
  FileText,
  Globe,
  DollarSign,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  BarChart3,
  Play,
} from "lucide-react";
import Markdown from "react-markdown";

export default function InvestmentResearchUI() {
  const [query, setQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const resultsRef = useRef(null);

  const sampleQueries = [
    "Compare NVIDIA vs AMD data center revenue growth and recent performance",
    "What is Microsoft's current stock price and latest news?",
    "Analyze NVIDIA: financials, news sentiment, and market performance",
    "What are AMD's main competitive risks according to latest filings?",
  ];

  const companies = [
    { ticker: "NVDA", name: "NVIDIA", color: "from-green-500 to-emerald-600" },
    { ticker: "AMD", name: "AMD", color: "from-red-500 to-rose-600" },
    { ticker: "MSFT", name: "Microsoft", color: "from-blue-500 to-cyan-600" },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || isProcessing) return;

    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("https://investment-research-bot-production.up.railway.app/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query }),
      });

      if (!response.ok) throw new Error("Failed to process query");

      const data = await response.json();
      setResult(data);

      // Scroll to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (err) {
      setError(err.message || "An error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  const ToolBadge = ({ tool }) => {
    const config = {
      RAG: {
        icon: FileText,
        color: "bg-blue-500/10 text-blue-400 border-blue-500/30",
      },
      WEB: {
        icon: Globe,
        color: "bg-purple-500/10 text-purple-400 border-purple-500/30",
      },
      STOCK: {
        icon: DollarSign,
        color: "bg-green-500/10 text-green-400 border-green-500/30",
      },
    };

    const { icon: Icon, color } = config[tool] || config.RAG;

    return (
      <span
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${color}`}
      >
        <Icon className="w-3 h-3" />
        {tool}
      </span>
    );
  };

  const markdownContent =
    typeof result?.finalOutput === "string"
      ? result.finalOutput.replace(/^`+|`+$/g, "").trim()
      : "";

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Investment Research AI
                </h1>
                <p className="text-xs text-slate-500">Multi-Agent RAG System</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-3 px-4 py-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-xs text-slate-400">6 Docs Indexed</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="max-w-4xl mx-auto text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm mb-6 animate-pulse">
            <Sparkles className="w-4 h-4" />
            <span>Powered by LangGraph + Multi-Hop RAG</span>
          </div>

          <h2 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
            <span className="bg-gradient-to-r from-slate-100 via-slate-200 to-slate-300 bg-clip-text text-transparent">
              Intelligent
            </span>
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Financial Analysis
            </span>
          </h2>

          <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-12">
            Ask complex investment questions. Get answers backed by SEC filings,
            real-time news, and market data.
          </p>

          {/* Search Bar */}
          <div className="relative mb-8">
            <div className="relative group">
              <Search className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500 group-focus-within:text-blue-400 transition-colors" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit(e)}
                placeholder="e.g., Compare NVIDIA vs AMD revenue growth and stock performance..."
                disabled={isProcessing}
                className="w-full pl-14 pr-32 py-6 bg-slate-900/50 border-2 border-slate-800 rounded-2xl text-lg focus:outline-none focus:border-blue-500/50 transition-all placeholder:text-slate-600 backdrop-blur-xl disabled:opacity-50"
              />
              <button
                onClick={handleSubmit}
                disabled={!query.trim() || isProcessing}
                className="absolute right-3 top-1/2 -translate-y-1/2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl font-semibold hover:from-blue-500 hover:to-purple-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg shadow-blue-500/20"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Processing
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Analyze
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Sample Queries */}
          <div className="mb-12">
            <p className="text-sm text-slate-500 mb-4">Try these examples:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {sampleQueries.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(sample)}
                  disabled={isProcessing}
                  className="text-left p-4 bg-slate-900/30 border border-slate-800 rounded-xl hover:border-blue-500/50 transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <p className="text-sm text-slate-300 group-hover:text-blue-400 transition-colors">
                    {sample}
                  </p>
                </button>
              ))}
            </div>
          </div>

          {/* Companies */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {companies.map((company) => (
              <div
                key={company.ticker}
                className="p-6 bg-slate-900/30 border border-slate-800 rounded-xl group hover:border-slate-700 transition-all"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div
                    className={`w-12 h-12 bg-gradient-to-br ${company.color} rounded-xl flex items-center justify-center shadow-lg`}
                  >
                    <BarChart3 className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-slate-200">
                      {company.name}
                    </p>
                    <p className="text-xs text-slate-500">{company.ticker}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <FileText className="w-3 h-3" />
                  <span>10-K Filings (2024, 2025)</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Simple Loading Indicator */}
        {isProcessing && (
          <div className="max-w-4xl mx-auto mb-12">
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 backdrop-blur-xl">
              <div className="flex flex-col items-center justify-center gap-4">
                <Loader2 className="w-12 h-12 text-blue-400 animate-spin" />
                <div className="text-center">
                  <p className="text-lg font-semibold text-slate-200 mb-2">
                    Analyzing your question...
                  </p>
                  <p className="text-sm text-slate-500">
                    This may take a few moments while we search through SEC filings, news, and market data
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="max-w-4xl mx-auto mb-12">
            <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-red-400 mb-1">
                    Error Processing Query
                  </p>
                  <p className="text-sm text-red-300/80">{error}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div ref={resultsRef} className="max-w-6xl mx-auto">
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl backdrop-blur-xl overflow-hidden">
              {/* Results Header */}
              <div className="px-8 py-6 border-b border-slate-800 bg-slate-900/80">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center shadow-lg">
                      <CheckCircle2 className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-slate-200">
                        Analysis Complete
                      </h3>
                      <p className="text-sm text-slate-500">
                        {result.toolsExecuted?.length || 0} tools used ·{" "}
                        {result.evidence?.length || 0} evidence pieces
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {result.toolsExecuted?.map((tool, idx) => (
                      <ToolBadge key={idx} tool={tool} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Query Info */}
              <div className="px-8 py-6 border-b border-slate-800 bg-slate-900/30">
                <div className="flex items-start gap-3">
                  <Search className="w-5 h-5 text-slate-500 flex-shrink-0 mt-1" />
                  <div className="flex-1">
                    <p className="text-sm text-slate-500 mb-1">
                      Original Question
                    </p>
                    <p className="text-slate-200">{result.originalQuestion}</p>
                  </div>
                </div>
              </div>

              {/* Final Answer */}
              <div className="px-8 py-8">
                <div className="prose prose-invert prose-slate max-w-none">
                  <div className="text-slate-300 leading-relaxed whitespace-pre-wrap">
                    {markdownContent && <Markdown>{markdownContent}</Markdown>}
                  </div>
                </div>
              </div>

              {/* Evidence Details */}
              {result.evidence && result.evidence.length > 0 && (
                <div className="px-8 py-6 border-t border-slate-800 bg-slate-900/30">
                  <details className="group">
                    <summary className="cursor-pointer text-sm font-semibold text-slate-400 hover:text-slate-300 transition-colors list-none flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      View Evidence Sources ({result.evidence.length})
                      <span className="ml-auto text-xs">Click to expand</span>
                    </summary>
                    <div className="mt-4 space-y-4">
                      {result.evidence.map((ev, idx) => (
                        <div
                          key={idx}
                          className="p-4 bg-slate-800/30 rounded-xl border border-slate-700/50"
                        >
                          <div className="flex items-start gap-3 mb-3">
                            <ToolBadge tool={ev.tool} />
                            <p className="text-sm text-slate-400 flex-1">
                              {ev.question}
                            </p>
                          </div>
                          <p className="text-sm text-slate-300 mb-2">
                            {ev.answer}
                          </p>
                          {ev.sources && ev.sources.length > 0 && (
                            <div className="text-xs text-slate-500">
                              Sources:{" "}
                              {ev.sources
                                .map(
                                  (s) =>
                                    `${s.company} ${s.year}, Page ${s.page}`,
                                )
                                .join(" • ")}
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
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950/50 backdrop-blur-xl mt-20">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-sm text-slate-500">
              Built with LangGraph • LangChain.js • Cohere • Pinecone • Groq •
              Tavily • Yahoo Finance
            </div>
            <div className="flex items-center gap-4 text-sm text-slate-500">
              <a
                href="https://github.com/IbrahimKhan7208/investment-research-bot"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-slate-300 transition-colors"
              >
                GitHub
              </a>
              <span>•</span>
              <span>© 2025 Investment Research AI</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
import YahooFinance from "yahoo-finance2";

const yahooFinance = new YahooFinance();

export class StockTool {
  constructor() {}

  /**
   * Get current stock price and basic info
   */
  async getCurrentPrice(ticker) {
    try {
      // ✅ FIXED: Use quoteSummary instead of quote (more reliable)
      const result = await yahooFinance.quoteSummary(ticker, {
        modules: ["price", "summaryDetail"],
      });

      const price = result.price;
      const summary = result.summaryDetail;

      return {
        ticker: ticker,
        price: price.regularMarketPrice,
        currency: price.currency,
        change: price.regularMarketChange,
        changePercent: price.regularMarketChangePercent,
        volume: price.regularMarketVolume,
        marketCap: price.marketCap,
        previousClose: price.regularMarketPreviousClose,
      };
    } catch (error) {
      console.error(`Failed to fetch price for ${ticker}:`, error.message);
      return null;
    }
  }

  /**
   * Get historical price data (for performance analysis)
   */
  async getHistoricalData(ticker, period = "1y") {
    try {
      const endDate = new Date();
      const startDate = this._getPeriodStartDate(period);

      // ✅ FIXED: Use correct parameter names
      const result = await yahooFinance.chart(ticker, {
        period1: startDate,
        period2: endDate,
        interval: "1d",
      });

      const quotes = result.quotes;

      if (!quotes || quotes.length === 0) {
        return null;
      }

      return {
        ticker: ticker,
        period: period,
        data: quotes.map((day) => ({
          date: day.date,
          close: day.close,
          volume: day.volume,
        })),
        startPrice: quotes[0]?.close,
        endPrice: quotes[quotes.length - 1]?.close,
        priceChange: quotes[quotes.length - 1]?.close - quotes[0]?.close,
        percentChange: (
          ((quotes[quotes.length - 1]?.close - quotes[0]?.close) /
            quotes[0]?.close) *
          100
        ).toFixed(2),
      };
    } catch (error) {
      console.error(
        `Failed to fetch historical data for ${ticker}:`,
        error.message,
      );
      return null;
    }
  }

  /**
   * Answer stock-related questions
   */
  async answerQuestion(question, companies = [], traceOut = {}) {
    const tickers = companies.length > 0 ? await this._resolveTickers(companies) : this._extractTickers(question);
    traceOut.tickers = tickers;

    if (tickers.length === 0) return [];

    const queryType = this._determineQueryType(question);
    traceOut.queryType = queryType;

    const results = [];

    for (const ticker of tickers) {
      if (queryType === "price" || queryType === "general") {
        const priceData = await this.getCurrentPrice(ticker);
        if (priceData) results.push(priceData);
      }

      if (queryType === "performance" || queryType === "historical") {
        const period = this._extractPeriod(question);
        traceOut.period = period;
        const histData = await this.getHistoricalData(ticker, period);
        if (histData) results.push(histData);
      }
    }
    return results;
  }

  // ══════════════════════════════════════════════════════════════
  // HELPER METHODS
  // ══════════════════════════════════════════════════════════════

  _extractTickers(question) {
    const tickerMap = {
      nvidia: "NVDA",
      nvda: "NVDA",
      amd: "AMD",
      microsoft: "MSFT",
      msft: "MSFT",
    };

    const tickers = [];
    const lowerQ = question.toLowerCase();

    for (const [key, ticker] of Object.entries(tickerMap)) {
      if (lowerQ.includes(key)) {
        if (!tickers.includes(ticker)) {
          tickers.push(ticker);
        }
      }
    }

    return tickers;
  }

  _determineQueryType(question) {
    const lowerQ = question.toLowerCase();

    // Check for time-related keywords
    const hasTimePeriod = /\d+\s*(month|year|week)|last|past|over the/i.test(
      lowerQ,
    );

    // Historical performance query
    if (
      hasTimePeriod ||
      lowerQ.includes("performance") ||
      lowerQ.includes("return")
    ) {
      return "performance";
    }

    // Current price query
    if (
      lowerQ.includes("current") ||
      lowerQ.includes("now") ||
      lowerQ.includes("trading at")
    ) {
      return "price";
    }

    if (lowerQ.includes("volume") || lowerQ.includes("market cap")) {
      return "general";
    }

    return "general";
  }

  _extractPeriod(question) {
    const lowerQ = question.toLowerCase();

    if (lowerQ.includes("year") || lowerQ.includes("12 month")) return "1y";
    if (lowerQ.includes("6 month")) return "6mo";
    if (lowerQ.includes("3 month")) return "3mo";
    if (lowerQ.includes("month")) return "1mo";
    if (lowerQ.includes("week")) return "1wk";

    return "1y"; // Default
  }

  async _resolveTickers(companies) {
    const tickers = [];

    for (const company of companies) {
      try {
        const results = await yahooFinance.search(company);
        const match = results?.quotes?.find(
          (q) => q.quoteType === "EQUITY" && q.symbol,
        );

        if (match && !tickers.includes(match.symbol)) {
          tickers.push(match.symbol);
        }
      } catch (error) {
        console.error(`Failed to resolve ticker for "${company}":`, error.message);
      }
    }

    // If Yahoo's search came up empty for everything, fall back to the
    // old regex map on the raw company names — cheap safety net, not a real fix
    if (tickers.length === 0) {
      return this._extractTickers(companies.join(" "));
    }

    return tickers;
  }

  _getPeriodStartDate(period) {
    const now = new Date();
    const map = {
      "1wk": 7,
      "1mo": 30,
      "3mo": 90,
      "6mo": 180,
      "1y": 365,
    };

    const days = map[period] || 365;
    const startDate = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);

    return startDate;
  }
}

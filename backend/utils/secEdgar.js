import "dotenv/config";
const USER_AGENT = process.env.SEC_USER_AGENT; // e.g. "InvestmentResearchBot/1.0 (you@example.com)" — SEC blocks anonymous requests
if (!USER_AGENT) {
  throw new Error("SEC_USER_AGENT env var is required");
}

let tickerMapCache = null;

async function loadTickerMap() {
  if (tickerMapCache) return tickerMapCache;

  const res = await fetch("https://www.sec.gov/files/company_tickers.json", {
    headers: { "User-Agent": USER_AGENT },
  });
  if (!res.ok) throw new Error(`Failed to load company_tickers.json: ${res.status}`);

  const data = await res.json();
  tickerMapCache = Object.values(data).map((entry) => ({
    cik: String(entry.cik_str).padStart(10, "0"),
    ticker: entry.ticker,
    title: entry.title,
  }));

  return tickerMapCache;
}

export async function resolveCik(companyNameOrTicker) {
  const map = await loadTickerMap();
  const query = companyNameOrTicker.trim().toLowerCase();

  let match = map.find((e) => e.ticker.toLowerCase() === query);
  if (!match) match = map.find((e) => e.title.toLowerCase() === query);
  if (!match) match = map.find((e) => e.title.toLowerCase().includes(query));

  return match || null; // { cik, ticker, title } or null
}

export async function getRecentFilings(cik, formType = "10-K", limit = 5) {
  const res = await fetch(`https://data.sec.gov/submissions/CIK${cik}.json`, {
    headers: { "User-Agent": USER_AGENT },
  });
  if (!res.ok) throw new Error(`Failed to load submissions for CIK ${cik}: ${res.status}`);

  const data = await res.json();
  const recent = data.filings.recent;

  const filings = [];
  for (let i = 0; i < recent.form.length; i++) {
    if (recent.form[i] !== formType) continue;

    filings.push({
      accessionNumber: recent.accessionNumber[i],
      filingDate: recent.filingDate[i],
      periodOfReport: recent.reportDate[i],
      primaryDocument: recent.primaryDocument[i],
    });

    if (filings.length >= limit) break;
  }

  return filings;
}

export function buildFilingUrl(cik, accessionNumber, primaryDocument) {
  const accessionNoDashes = accessionNumber.replace(/-/g, "");
  const cikNoLeadingZeros = String(Number(cik));
  return `https://www.sec.gov/Archives/edgar/data/${cikNoLeadingZeros}/${accessionNoDashes}/${primaryDocument}`;
}

export async function fetchFilingHtml(url) {
  const res = await fetch(url, { headers: { "User-Agent": USER_AGENT } });
  if (!res.ok) throw new Error(`Failed to fetch filing ${url}: ${res.status}`);
  return res.text();
}
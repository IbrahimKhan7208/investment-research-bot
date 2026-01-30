import { StockTool } from './utils/stockTool.js';

async function test() {
  const stockTool = new StockTool();
  
  console.log("Testing NVDA current price...");
  const price = await stockTool.getCurrentPrice('NVDA');
  console.log(price);
  
  console.log("\nTesting NVDA historical (1 year)...");
  const hist = await stockTool.getHistoricalData('NVDA', '1y');
  console.log({
    ticker: hist.ticker,
    startPrice: hist.startPrice,
    endPrice: hist.endPrice,
    percentChange: hist.percentChange
  });
  
  console.log("\nTesting answerQuestion...");
  const answer = await stockTool.answerQuestion("What is NVIDIA's current stock price?");
  console.log(answer);
}

test();
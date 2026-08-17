import { routeInvestmentQuestion } from "./routerModel.js";

const routing = await routeInvestmentQuestion({
  question: "Compare NVIDIA vs AMD's data center revenue growth, recent news sentiment, and stock performance",
});

console.log(JSON.stringify(routing, null, 2));
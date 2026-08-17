import * as cheerio from "cheerio";
import TurndownService from "turndown";
import { tableToProse } from "./tableToProse.js";

function stripChrome($) {
  $("ix\\:header, ix\\:hidden, ix\\:references, ix\\:resources").remove();
  $('[style*="display:none"], [style*="display: none"]').remove();
  $(".xbrl-non-visible-content").remove();

  $("*").each((_, el) => {
    const $el = $(el);
    if ($el.children().length === 0 && $el.text().trim() === "Table of Contents") {
      $el.remove();
    }
  });

  return $;
}

export function filingHtmlToMarkdown(html) {
  const $ = stripChrome(cheerio.load(html));

  // Sentinel markers deliberately have NO underscore — Turndown escapes
  // literal "_" in text content (it's markdown italic syntax) to "\_",
  // which silently breaks any exact-string match against the marker later
  // in chunkFiling.js. Confirmed via repro: with underscores, 0/64 real
  // tables from an NVDA 10-K survived this conversion; without them, 64/64 did.
  $("table").each((_, table) => {
    const prose = tableToProse($, table);
    $(table).replaceWith(`<p>§TABLEPROSESTART§${prose.replace(/\n/g, "<br/>")}§TABLEPROSEEND§</p>`);
  });

  const turndown = new TurndownService({ headingStyle: "atx", codeBlockStyle: "fenced" });
  return turndown.turndown($.html());
}
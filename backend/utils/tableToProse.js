function isNumericLike(v) {
  if (v === "$") return true;
  return /^\(?-?\$?[\d,]+(\.\d+)?%?\)?$/.test(v);
}

function tableToGrid($, table) {
  const $rows = $(table).find("tr");
  const grid = [];
  const rowSpans = {};

  $rows.each((rowIndex, tr) => {
    grid[rowIndex] = grid[rowIndex] || [];
    let colIndex = 0;
    const cells = $(tr).find("td, th").toArray();
    let cellPointer = 0;

    while (cellPointer < cells.length || rowSpans[colIndex]) {
      if (rowSpans[colIndex] && rowSpans[colIndex].remaining > 0) {
        grid[rowIndex][colIndex] = { value: rowSpans[colIndex].value, isSpanFill: true };
        rowSpans[colIndex].remaining--;
        if (rowSpans[colIndex].remaining === 0) delete rowSpans[colIndex];
        colIndex++;
        continue;
      }
      if (cellPointer >= cells.length) break;

      const $cell = $(cells[cellPointer]);
      const value = $cell.text().trim();
      const colspan = parseInt($cell.attr("colspan") || "1", 10);
      const rowspan = parseInt($cell.attr("rowspan") || "1", 10);

      for (let i = 0; i < colspan; i++) {
        grid[rowIndex][colIndex] = { value, isSpanFill: i > 0 };
        if (rowspan > 1) rowSpans[colIndex] = { remaining: rowspan - 1, value };
        colIndex++;
      }
      cellPointer++;
    }
  });

  return grid;
}

function splitHeaderRows(grid) {
  for (let r = 0; r < grid.length; r++) {
    const row = grid[r];
    const hasLabel = row[0] && row[0].value && !isNumericLike(row[0].value);
    const hasNumeric = row.slice(1).some((c) => c && c.value && isNumericLike(c.value));
    if (hasLabel && hasNumeric) {
      return { headerRows: grid.slice(0, r), dataRows: grid.slice(r) };
    }
  }
  return { headerRows: [], dataRows: grid };
}

function mergeSpacerColumns(headerRows, dataRows) {
  const numCols = Math.max(...headerRows.map((r) => r.length), ...dataRows.map((r) => r.length), 0);
  const spacerCols = new Set();

  for (let c = 0; c < numCols; c++) {
    const allBlankOrDollar = dataRows.every((row) => {
      const v = row[c] ? row[c].value : "";
      return v === "" || v === "$";
    });
    if (allBlankOrDollar) spacerCols.add(c);
  }

  const mergeRow = (row, { withDollarCarry }) => {
    const merged = [];
    let carry = "";
    for (let c = 0; c < numCols; c++) {
      const cell = row[c] || { value: "" };
      if (spacerCols.has(c)) {
        if (withDollarCarry && cell.value === "$") carry += "$";
        else if (!withDollarCarry && cell.value && cell.value !== carry) carry = cell.value;
        continue;
      }
      const value = withDollarCarry
        ? carry + cell.value
        : cell.value === carry ? cell.value : [carry, cell.value].filter(Boolean).join(" ");
      merged.push({ value, isSpanFill: cell.isSpanFill });
      carry = "";
    }
    return merged;
  };

  return {
    headerRows: headerRows.map((row) => mergeRow(row, { withDollarCarry: false })),
    dataRows: dataRows.map((row) => mergeRow(row, { withDollarCarry: true })),
  };
}

function buildColumnLabels(headerRows, numCols) {
  const labels = [];
  for (let c = 1; c < numCols; c++) {
    const parts = [];
    for (const row of headerRows) {
      const v = row[c] && !row[c].isSpanFill ? row[c].value : "";
      if (v && parts[parts.length - 1] !== v) parts.push(v);
    }
    labels.push(parts.join(" "));
  }
  return labels;
}

export function tableToProse($, table) {
  const rawGrid = tableToGrid($, table);
  if (rawGrid.length === 0) return "";

  const split = splitHeaderRows(rawGrid);
  const { headerRows, dataRows } = mergeSpacerColumns(split.headerRows, split.dataRows);
  const numCols = Math.max(...headerRows.map((r) => r.length), ...dataRows.map((r) => r.length));
  const colLabels = buildColumnLabels(headerRows, numCols);

  const lines = [];
  for (const row of dataRows) {
    const rowLabel = row[0] ? row[0].value : "";
    if (!rowLabel) continue;

    const parts = [];
    for (let c = 1; c < numCols; c++) {
      const v = row[c] && !row[c].isSpanFill ? row[c].value : "";
      if (!v) continue;
      const label = colLabels[c - 1] || `col${c}`;
      parts.push(`${label}: ${v}`);
    }

    lines.push(parts.length > 0 ? `${rowLabel} — ${parts.join("; ")}.` : `${rowLabel}.`);
  }

  return lines.join("\n");
}
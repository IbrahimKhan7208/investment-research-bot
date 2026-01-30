export async function extractMetadata(filename) {
  const withoutExtension = filename.replace(".pdf", " ");
  const parts = withoutExtension.split("-");
  const company = parts[0];
  const year = parseInt(parts[1]);
  // console.log(company, year);
  return {
    company: company,
    year: year,
    source: filename,
  };
}

// const meta = extractMetadata("NVIDIA-2024.pdf");
// console.log(meta);
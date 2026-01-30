import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export async function chunkPDF(filePath, baseMetadata) {
  const loader = new PDFLoader(filePath, { splitPages: true });

  const pages = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 })
  // console.log(pages);
  const allChunks = []

  for ( let pageIndex = 0; pageIndex < pages.length; pageIndex++){
    const page = pages[pageIndex]
    const pageNumber = pageIndex + 1
    const pageChunks = await splitter.splitText(page.pageContent)

    for (const chunkText of pageChunks){
      allChunks.push({
        pageContent: chunkText,
        metadata: {
          ...baseMetadata,
          page: pageNumber
        }
      })
    }
  }
  return allChunks
  // console.log(pages[0]);
  // console.log(allChunks);
}

// const chunks = await chunkPDF("../reports/AMD-2024.pdf")
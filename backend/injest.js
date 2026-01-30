import dotenv from "dotenv";
dotenv.config();
import { CohereEmbeddings } from "@langchain/cohere";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { extractMetadata } from "./utils/extractMetadata.js";
import path from "path";
import { chunkPDF } from "./utils/chunkPDF.js";

const PDF_FILES = [
  "AMD-2024.pdf",
  "AMD-2025.pdf",
  "Microsoft-2024.pdf",
  "Microsoft-2025.pdf",
  "NVIDIA-2024.pdf",
  "NVIDIA-2025.pdf",
];

async function injestDocuments() {
  const embeddings = new CohereEmbeddings({
     model: "embed-english-v3.0",
     apiKey: process.env.COHERE_API_KEY,
   });

  const pinecone = new PineconeClient();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });

  for (const filename of PDF_FILES) {
    const baseMetadata = await extractMetadata(filename);
    const filePath = path.join("./reports", filename);
    const chunks = await chunkPDF(filePath, baseMetadata);
    console.log(chunks.length);

    await vectorStore.addDocuments(chunks);
  }
  console.log("INJESTED!");
}

await injestDocuments();

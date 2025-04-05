//@ts-nocheck
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { Embeddings } from "@langchain/core/embeddings";
import { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { BaseChatModelCallOptions } from "@langchain/core/language_models/chat_models";

// Custom wrapper to log embeddings
class LoggingEmbeddingsWrapper implements Embeddings {
  private embeddings: Embeddings;
  private name: string;
  caller: string = "LoggingEmbeddingsWrapper"; // Add the caller property

  constructor(embeddings: Embeddings, name: string) {
    this.embeddings = embeddings;
    this.name = name;
  }

  async embedQuery(text: string): Promise<number[]> {
    console.log(`[${this.name}] Embedding query: "${text}"`);
    const embeddings = await this.embeddings.embedQuery(text);
    console.log(`[${this.name}] Embedding dimensions: ${embeddings.length}`);
    console.log(`[${this.name}] First 5 values: [${embeddings.slice(0, 5).join(', ')}]`);
    return embeddings;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    console.log(`[${this.name}] Embedding ${documents.length} documents`);
    if (documents.length > 0) {
      console.log(`[${this.name}] First document preview: "${documents[0].substring(0, 50)}..."`);
    }
    const embeddings = await this.embeddings.embedDocuments(documents);
    console.log(`[${this.name}] Embeddings created with dimensions: ${embeddings[0]?.length || 0}`);
    return embeddings;
  }
}

// Helper function to get the chat model based on environment variables
export function getChatModel(
    temperature: number = 0.8,
    modelName?: string,
    // Allow passing additional options, but be cautious as they might be model-specific
    options?: Partial<BaseChatModelCallOptions>
): BaseChatModel {
  if (process.env.GOOGLE_API_KEY) {
    console.log("Using Google Generative AI Model");
    // Ensure GOOGLE_API_KEY is non-empty
    if (!process.env.GOOGLE_API_KEY) {
        throw new Error("GOOGLE_API_KEY is set but empty.");
    }
    return new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      model: modelName || "gemini-1.5-flash", // Default to a known Gemini model
      temperature: temperature,
      ...options,
      // Note: Gemini-specific features like structured output/tool calling
      // might require different setup or parameters than OpenAI.
      // The `withStructuredOutput` and agent tool calling helpers used
      // elsewhere might not work directly with Gemini without adjustments.
    });
  } else if (process.env.OPENAI_API_KEY) {
    console.log("Using OpenAI Model");
     // Ensure OPENAI_API_KEY is non-empty
    if (!process.env.OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is set but empty.");
    }
    return new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      temperature: temperature,
      model: modelName || "gpt-4o-mini", // Keep existing default
      ...options,
    });
  } else {
    throw new Error("No API key found for OpenAI or Google Generative AI. Please set OPENAI_API_KEY or GOOGLE_API_KEY in your .env.local file.");
  }
}

// Helper function to get embeddings based on environment variables
export function getEmbeddings(p0?: { model: string; dimensions: number; }): Embeddings {
  if (process.env.GOOGLE_API_KEY) {
    console.log("Using Google Generative AI Embeddings");
    // Ensure GOOGLE_API_KEY is non-empty
    if (!process.env.GOOGLE_API_KEY) {
      throw new Error("GOOGLE_API_KEY is set but empty.");
    }
    const baseEmbeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "embedding-001", // Default Gemini embedding model
    });
    // @ts-ignore'
    return new LoggingEmbeddingsWrapper(baseEmbeddings, "GoogleGenerativeAIEmbeddings");
  }  else {
    throw new Error("No API key found for OpenAI or Google Generative AI Embeddings. Please set OPENAI_API_KEY or GOOGLE_API_KEY in your .env.local file.");
  }
}

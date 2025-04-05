import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage, StreamingTextResponse } from "ai";

import { createClient } from "@supabase/supabase-js";

import { PromptTemplate } from "@langchain/core/prompts";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { Document } from "@langchain/core/documents";
import { RunnableSequence } from "@langchain/core/runnables";
import {
  BytesOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { getChatModel, getEmbeddings } from "@/utils/modelSelection";

export const runtime = "edge";

const combineDocumentsFn = (docs: Document[]) => {
  const serializedDocs = docs.map((doc) => doc.pageContent);
  return serializedDocs.join("\n\n");
};

const formatVercelMessages = (chatHistory: VercelChatMessage[]) => {
  const formattedDialogueTurns = chatHistory.map((message) => {
    if (message.role === "user") {
      return `Human: ${message.content}`;
    } else if (message.role === "assistant") {
      return `Assistant: ${message.content}`;
    } else {
      return `${message.role}: ${message.content}`;
    }
  });
  return formattedDialogueTurns.join("\n");
};

const CONDENSE_QUESTION_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;
const condenseQuestionPrompt = PromptTemplate.fromTemplate(
  CONDENSE_QUESTION_TEMPLATE,
);

const ANSWER_TEMPLATE = `Answer the question based only on the following context:
{context}

Question: {question}

If the context doesn't directly mention "GymNavigator", you can still provide information about gym management systems based on what's available in the context. 
Make sure your answer is helpful and indicates it's based on the information provided.
`;
const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE);

/**
 * This handler initializes and calls a retrieval chain. It composes the chain using
 * LangChain Expression Language. See the docs for more information:
 *
 * https://js.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const previousMessages = messages.slice(0, -1);
    const currentMessageContent = messages[messages.length - 1].content;

    /**
     * Use utility functions to get the model and embeddings.
     * Updated to use the Geminis embedding model.
     */
    const model = getChatModel(0.2, undefined);
    const embeddings = getEmbeddings({ model: "embedding-001", dimensions: 768 });
    console.log("Retrieval Route: Using Geminis embedding model 'embedding-001' with dimensions 768");

    const client = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_PRIVATE_KEY!,
    );
    
    // Create vector store without queryName for diagnostic purposes
    const vectorstore = new SupabaseVectorStore(embeddings, {
      client,
      tableName: "documents",
    });

    // Added: Check document count in the "documents" table
    const { count, error } = await client
      .from("documents")
      .select("*", { count: "exact", head: true });
    if (error) {
      console.error("Error counting docs:", error);
    } else {
      console.log(`Document table count: ${count}`);
    }

    // Query information_schema.routines to check if "match_documents" exists
    const { data: routineData, error: routineError } = await client
      .from("information_schema.routines")
      .select("*")
      .eq("routine_name", "match_documents");

    if (routineError) {
      console.error("Error checking stored procedure 'match_documents':", routineError);
    } else if (routineData && routineData.length > 0) {
      console.log("Stored procedure 'match_documents' exists:", routineData);
    } else {
      console.log("Stored procedure 'match_documents' does not exist.");
    }

    // Execute only the final SQL query implementation which successfully retrieves documents.
    let finalDocuments: Document[] = [];
    const { data: finalSqlResults, error: finalSqlError } = await client
      .from('documents')
      .select('*')
      .limit(3);
    
    if (finalSqlError) {
      console.error("FINAL SQL ERROR:", finalSqlError);
      throw new Error(finalSqlError.message);
    } else {
      console.log(`FINAL SQL: Successfully retrieved ${finalSqlResults?.length || 0} documents`);
      if (finalSqlResults && finalSqlResults.length > 0) {
        finalDocuments = finalSqlResults.map((item) => new Document({
          pageContent: item.content,
          metadata: item.metadata || {}
        }));
      }
    }

    /**
     * We use LangChain Expression Language to compose two chains.
     * To learn more, see the guide here:
     *
     * https://js.langchain.com/docs/guides/expression_language/cookbook
     *
     * You can also use the "createRetrievalChain" method with a
     * "historyAwareRetriever" to get something prebaked.
     */
    const standaloneQuestionChain = RunnableSequence.from([
      condenseQuestionPrompt,
      model,
      new StringOutputParser(),
    ]);

    // Log the standalone question
    standaloneQuestionChain.invoke({
      chat_history: formatVercelMessages(previousMessages),
      question: currentMessageContent,
    }).then(standaloneQuestion => {
      console.log("Retrieval Route: Standalone question for retrieval:", standaloneQuestion);
      
      // Also log that this standalone question will be used for embeddings
      console.log("Retrieval Route: This standalone question will be embedded using Geminis model");
    });

    const retriever = vectorstore.asRetriever({
      callbacks: [
        {
          handleRetrieverStart(retriever, documents, runId, parentRunId, tags, metadata) {
            console.log("Retrieval Route: Retriever starting with query:", documents);
            console.log("Retrieval Route: Vector store config:", {
              tableName: vectorstore.tableName,
              queryName: vectorstore.queryName
            });
          },
          handleRetrieverEnd(documents) {
            console.log(`Retrieval Route: Retrieved ${documents.length} documents.`);
            if (documents.length === 0) {
              console.log("Retrieval Route: No documents found in vector store");
            } else {
              documents.forEach((doc, index) => {
                console.log(`Retrieval Route: Doc ${index + 1}:`, {
                  content: doc.pageContent.substring(0, 100) + "...",
                  metadata: doc.metadata,
                });
              });
            }
          },
        },
      ],
    });

    const retrievalChain = retriever.pipe(combineDocumentsFn);

    const answerChain = RunnableSequence.from([
      {
        context: RunnableSequence.from([
          (input) => input.question,
          retrievalChain,
        ]),
        chat_history: (input) => input.chat_history,
        question: (input) => input.question,
      },
      answerPrompt,
      model,
    ]);

    const conversationalRetrievalQAChain = RunnableSequence.from([
      {
        question: standaloneQuestionChain,
        chat_history: (input) => input.chat_history,
      },
      answerChain,
      new BytesOutputParser(),
    ]);

    const stream = await conversationalRetrievalQAChain.stream({
      question: currentMessageContent,
      chat_history: formatVercelMessages(previousMessages),
    });

    return new StreamingTextResponse(stream, {
      headers: {
        "x-message-index": (previousMessages.length + 1).toString(),
        "x-sources": Buffer.from(JSON.stringify(
          finalDocuments.map((doc) => ({
            pageContent: doc.pageContent.slice(0, 50) + "...",
            metadata: doc.metadata,
          }))
        )).toString("base64"),
      },
    });
  } catch (e: any) {
    console.error("Chat API error:", e);
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}

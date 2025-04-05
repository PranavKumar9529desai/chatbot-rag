import { NextRequest, NextResponse } from "next/server";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { getEmbeddings } from "@/utils/modelSelection"; // Import the utility function

export const runtime = "edge";

// Before running, follow set-up instructions at
// https://js.langchain.com/v0.2/docs/integrations/vectorstores/supabase

/**
 * This handler takes input text, splits it into chunks, and embeds those chunks
 * into a vector store for later retrieval. See the following docs for more information:
 *
 * https://js.langchain.com/v0.2/docs/how_to/recursive_text_splitter
 * https://js.langchain.com/v0.2/docs/integrations/vectorstores/supabase
 */
export async function POST(req: NextRequest) {
  const body = await req.json();
  const text = body.text;

  if (process.env.NEXT_PUBLIC_DEMO === "true") {
    return NextResponse.json(
      {
        error: [
          "Ingest is not supported in demo mode.",
          "Please set up your own version of the repo here: https://github.com/langchain-ai/langchain-nextjs-template",
        ].join("\n"),
      },
      { status: 403 },
    );
  }

  try {
    const client = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_PRIVATE_KEY!,
    );

    const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
      chunkSize: 256,
      chunkOverlap: 20,
    });

    console.log("Ingest: Splitting document...");
    const splitDocuments = await splitter.createDocuments([text]);
    console.log(`Ingest: Created ${splitDocuments.length} split documents.`);
    if (splitDocuments.length === 0) {
      console.warn("Ingest: No documents were created after splitting. Check input text and splitter settings.");
      // Optionally return an error or specific response if no documents are generated
      // return NextResponse.json({ error: "No documents to ingest after splitting." }, { status: 400 });
    } else {
       // Log the first document's content snippet for verification
       console.log("Ingest: First split document content snippet:", splitDocuments[0].pageContent.substring(0, 100) + "...");
    }

    /**
     * Use the utility function to get the appropriate embeddings.
     * Updated to use the Geminis embedding model.
     */
    const embeddings = getEmbeddings({ model: "embedding-001", dimensions: 768 });
    console.log("Ingest: Using Geminis embedding model 'embedding-001' with dimensions 768");

    console.log("Ingest: Attempting to add documents to Supabase Vector Store...");
    const vectorstore = await SupabaseVectorStore.fromDocuments(
      splitDocuments,
      embeddings, // Use the selected embeddings
      {
        client,
        tableName: "documents",
        queryName: "match_documents",
      },
    );
    console.log("Ingest: SupabaseVectorStore.fromDocuments call completed.");

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (e: any) {
    console.error(e);
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}

import { NextRequest, NextResponse } from "next/server";
import { Message as VercelChatMessage, StreamingTextResponse } from "ai";

import { PromptTemplate } from "@langchain/core/prompts";
// Correct import path for StringOutputParser
import { StringOutputParser } from "@langchain/core/output_parsers"; 
import { HttpResponseOutputParser } from "langchain/output_parsers";
import { getChatModel } from "@/utils/modelSelection";

export const runtime = "edge";

const formatMessage = (message: VercelChatMessage) => {
  return `${message.role}: ${message.content}`;
};

const TEMPLATE = `You are a helpful assistant.

Current conversation:
{chat_history}

User: {input}
AI:`;

/**
 * This handler initializes and calls a simple chain with a prompt,
 * chat model, and output parser. See the docs for more information:
 *
 * https://js.langchain.com/docs/guides/expression_language/cookbook#prompttemplate--llm--outputparser
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);
    const currentMessageContent = messages[messages.length - 1].content;
    const prompt = PromptTemplate.fromTemplate(TEMPLATE);

    /**
     * Use the utility function to get the appropriate chat model.
     */
    const model = getChatModel(0.8, undefined);

    /**
     * Create a chain that pipes the prompt to the model, and then to the output parser.
     * Use StringOutputParser as a fallback for Gemini API compatibility
     */
    const isUsingGemini = !!process.env.GOOGLE_API_KEY;
    console.log("isUsingGemini", isUsingGemini);
    const outputParser = isUsingGemini 
      ? new StringOutputParser()
      : new HttpResponseOutputParser();
    // @ts-ignore
    const chain = prompt.pipe(model).pipe(outputParser);

    const stream = await chain.stream({
      chat_history: formattedPreviousMessages.join("\n"),
      input: currentMessageContent,
    });

    return new StreamingTextResponse(stream);
  } catch (e: any) {
    console.error("Chat API error:", e);
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}

import { NextRequest, NextResponse } from "next/server";

import { z } from "zod";

import { PromptTemplate } from "@langchain/core/prompts";
import { getChatModel } from "@/utils/modelSelection"; // Import the utility function

export const runtime = "edge";

const TEMPLATE = `Extract the requested fields from the input.

The field "entity" refers to the first mentioned entity in the input.

Input:

{input}`;

/**
 * This handler initializes and calls a structured output chain.
 * See the docs for more information:
 *
 * https://js.langchain.com/v0.2/docs/how_to/structured_output
 * NOTE: Gemini models might require different setup for structured output/function calling.
 * The .withStructuredOutput() method is primarily designed for OpenAI.
 * Adjustments may be needed if using GOOGLE_API_KEY.
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const messages = body.messages ?? [];
    const currentMessageContent = messages[messages.length - 1].content;

    const prompt = PromptTemplate.fromTemplate(TEMPLATE);
    /**
     * Use the utility function to get the appropriate chat model.
     * Function calling/structured output is generally model-specific.
     */
    const model = getChatModel(0.8, undefined);

    /**
     * We use Zod (https://zod.dev) to define our schema for convenience,
     * but you can pass JSON schema if desired.
     */
    const schema = z
      .object({
        tone: z
          .enum(["positive", "negative", "neutral"])
          .describe("The overall tone of the input"),
        entity: z.string().describe("The entity mentioned in the input"),
        word_count: z.number().describe("The number of words in the input"),
        chat_response: z.string().describe("A response to the human's input"),
        final_punctuation: z
          .optional(z.string())
          .describe("The final punctuation mark in the input, if any."),
      })
      .describe("Should always be used to properly format output");

    /**
     * Bind schema to the model.
     * NOTE: This .withStructuredOutput method is heavily optimized for OpenAI
     * and may not work as expected with Gemini models without adjustments
     * or using Gemini-specific function calling methods.
     */
    const structuredOutputModel = model.withStructuredOutput(schema, {
      name: "output_formatter",
    });

    /**
     * Returns a chain with the function calling model.
     */
    const chain = prompt.pipe(structuredOutputModel);

    const result = await chain.invoke({
      input: currentMessageContent,
    });

    return NextResponse.json(result, { status: 200 });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: e.status ?? 500 });
  }
}

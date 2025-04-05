"use server";

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStreamableValue } from "ai/rsc";
import { z } from "zod";
import { Runnable } from "@langchain/core/runnables";
import { zodToJsonSchema } from "zod-to-json-schema";
import { getChatModel } from "@/utils/modelSelection"; // Import the utility function
import { StructuredOutputParser } from "langchain/output_parsers";

const Weather = z
  .object({
    city: z.string().describe("City to search for weather"),
    state: z.string().describe("State abbreviation to search for weather"),
  })
  .describe("Weather search parameters");

/**
 * NOTE: Tool calling/structured output behavior can be model-specific.
 * The setup here (binding tools, using JsonOutputKeyToolsParser, withStructuredOutput)
 * is heavily based on OpenAI's function/tool calling.
 * Adjustments may be needed for Gemini models.
 */
export async function executeTool(
  input: string,
  options?: {
    wso?: boolean;
    streamEvents?: boolean; // Note: streamEvents option might not be directly applicable/supported with Gemini in the same way
  },
) {
  "use server";

  const stream = createStreamableValue();

  (async () => {
    let chain: Runnable;

    if (options?.wso) {
      const prompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          `You are a helpful assistant. Use the tools provided to best assist the user.`,
        ],
        ["human", "{input}"],
      ]);

      const llm = getChatModel(0, undefined);

      chain = prompt.pipe(
        llm.withStructuredOutput(Weather, {
          name: "get_weather",
        }),
      );
    } else {
      if (process.env.GOOGLE_API_KEY) {
        // Gemini Tool Calling
        const parser = StructuredOutputParser.fromZodSchema(Weather);
        const formatInstructions = parser.getFormatInstructions();

        const prompt = ChatPromptTemplate.fromMessages([
          [
            "system",
            `You are a helpful assistant. Use the tools provided to best assist the user. To get the weather, respond with the following JSON:\n\n${formatInstructions}`,
          ],
          ["human", "{input}"],
        ]);

        const llm = getChatModel(0, undefined);

        chain = prompt.pipe(llm).pipe(parser);
      } else {
        chain = ChatPromptTemplate.fromMessages([
          [
            "system",
            `You are a helpful assistant. Use the tools provided to best assist the user.`,
          ],
          ["human", "{input}"],
        ]).pipe(getChatModel(0, undefined));
      }
    }

    const streamResult = await chain.stream({
      input,
    });

    for await (const item of streamResult) {
      // Event structure might differ
      stream.update(JSON.parse(JSON.stringify(item, null, 2)));
    }

    stream.done();
  })();

  return { streamData: stream.value };
}

"use server";

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStreamableValue } from "ai/rsc";
import { z } from "zod";
import { Runnable } from "@langchain/core/runnables";
import { zodToJsonSchema } from "zod-to-json-schema";
import { JsonOutputKeyToolsParser } from "@langchain/core/output_parsers/openai_tools";
import { getChatModel } from "@/utils/modelSelection"; // Import the utility function

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
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a helpful assistant. Use the tools provided to best assist the user.`,
      ],
      ["human", "{input}"],
    ]);

    /**
     * Use the utility function to get the appropriate chat model.
     */
    const llm = getChatModel(0, undefined);

    let chain: Runnable;

    if (options?.wso) {
      /**
       * NOTE: withStructuredOutput is primarily for OpenAI.
       * This branch might not work correctly with Gemini without adjustments.
       */
      chain = prompt.pipe(
        llm.withStructuredOutput(Weather, {
          name: "get_weather",
        }),
      );
    } else {
      /**
       * NOTE: Binding tools like this is standard for OpenAI.
       * Gemini might require a different approach for tool integration.
       * JsonOutputKeyToolsParser is also specific to OpenAI tool parsing.
       */
      chain = prompt
        .pipe(
          // @ts-ignore - Binding OpenAI specific tool parameters to the base model type
          llm.bind({
            tools: [
              {
                type: "function" as const,
                function: {
                  name: "get_weather",
                  description: Weather.description,
                  parameters: zodToJsonSchema(Weather),
                },
              },
            ],
            tool_choice: "get_weather", // Forcing tool choice might behave differently
          }),
        )
        .pipe(
          new JsonOutputKeyToolsParser<z.infer<typeof Weather>>({
            keyName: "get_weather",
            zodSchema: Weather,
          }),
        );
    }

    const streamResult = await chain.stream({ // streamEvents might not be applicable here depending on chain type
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

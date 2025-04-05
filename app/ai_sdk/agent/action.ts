"use server";

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { pull } from "langchain/hub";
import { createStreamableValue } from "ai/rsc";
import { getChatModel } from "@/utils/modelSelection"; // Import the utility function

/**
 * NOTE: Agent behavior, especially tool calling, can be model-specific.
 * The agent setup (prompt, tool calling agent creation) might require
 * adjustments for Gemini models.
 */
export async function runAgent(input: string) {
  "use server";

  const stream = createStreamableValue();
  (async () => {
    const tools = [new TavilySearchResults({ maxResults: 1 })];
    // This prompt is specifically for OpenAI tools agent.
    // May need a different prompt or agent setup for Gemini.
    const prompt = await pull<ChatPromptTemplate>(
      "hwchase17/openai-tools-agent",
    );

    /**
     * Use the utility function to get the appropriate chat model.
     */
    const llm = getChatModel(0, undefined);

    /**
     * NOTE: createToolCallingAgent is often optimized for OpenAI.
     * Verify compatibility or use Gemini-specific agent creation methods if needed.
     */
    const agent = createToolCallingAgent({
      llm,
      tools,
      prompt,
    });

    const agentExecutor = new AgentExecutor({ agent, tools });

    const streamingEvents = agentExecutor.streamEvents(
      { input },
      { version: "v2" },
    );

    for await (const item of streamingEvents) {
      // The structure of events might differ between models.
      stream.update(JSON.parse(JSON.stringify(item, null, 2)));
    }

    stream.done();
  })();

  return { streamData: stream.value };
}

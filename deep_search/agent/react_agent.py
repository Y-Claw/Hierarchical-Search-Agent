import os
import json
import math
import random
from contextlib import AsyncExitStack

from deep_search.agent.prompts.agent_prompting import get_full_system_prompt, ENGLISH_OPENAI_PROMPT, TOOL_SYSTEM_PROMPT, MOCK_QWQ_PROMPT, ENGLISH_MOCK_QWQ_PROMPT, ENGLISH_TOOL_SYSTEM_PROMPT, ENGLISH_SUMMARY_TOOL_SYSTEM_PROMPT, ENGLISH_QWQ_PROMPT, ENGLISH_SUMMARY_QWQ_PROMPT, ENGLISH_SINGLE_SEARCH_QWQ_PROMPT
from deep_search.agent.state_evaluator.information_evaluator import InformationEvaluator
from deep_search.utils.api import call_openai
from deep_search.utils.memory import ActionStep, ToolStep, BaseStep, StepManager, ObservationStep, NoticeStep, UserStep, SystemStep
from deep_search.utils.agent_utils import get_have_internet, current_datetime
from deep_search.utils.enums import QWEN_MODEL, QWQ_MODEL, OPENAI_MODEL, REACT_MODEL, FUNC_ONLY_MODEL
from deep_search.utils.config import DOWNLOAD_DIR

from search_toolkit.mcp.client import MCPClient

class ReactAgent:
    
    def __init__(self, model: str, toolcall_model: str=None, agent_work_dir: str=None, mcp_server_name: str=None, max_steps: int=16, 
                 expand_num: int=2, mcts_exploration_weight: float=1.0, evaluator_configs: dict=None, **kwargs):
        self.model = model
        self.toolcall_model = toolcall_model if toolcall_model else model
        self.agent_work_dir = agent_work_dir
        self.api_key = os.getenv('ONE_API_KEY')
        self.base_url = os.getenv('ONE_API_BASE')
        # self.have_internet = get_have_internet()
        self.date_str = current_datetime()
        self.max_steps = max_steps
        self.mcp_server_name = mcp_server_name
        self.mcp_client = None
        self.tools = None
        self.memory = StepManager(model=self.model, agent_work_dir=self.agent_work_dir)
        self.exit_stack = AsyncExitStack()
        self.exit_tool_calls = [{"type": "function", "function": {"name": "all_information_sufficient", "arguments": {}}}]
        self.mock_thought = kwargs.get("mock_thought", False)
        self.visit_type = kwargs.get("visit_type", "content")

    def set_mcp(self, mcp_client: MCPClient, tools: list[dict]):
        self.mcp_client = mcp_client
        self.tools = tools
    
    def _get_system_prompt(self, text_context_list: list[str]=None, image_file: str=None) -> list[dict]:
        if self.model in OPENAI_MODEL or self.model in QWEN_MODEL:
            system_prompt = ENGLISH_OPENAI_PROMPT
        elif self.model in QWQ_MODEL:
            if self.mock_thought:
                # system_prompt = MOCK_QWQ_PROMPT
                system_prompt = ENGLISH_MOCK_QWQ_PROMPT
            elif self.mcp_server_name == "single_search":
                import pdb; pdb.set_trace()
                system_prompt = ENGLISH_SINGLE_SEARCH_QWQ_PROMPT
            elif self.visit_type == "summary":
                system_prompt = ENGLISH_SUMMARY_QWQ_PROMPT
            else:
                # system_prompt = get_full_system_prompt(agent_work_dir=DOWNLOAD_DIR, text_context_list=text_context_list, image_file=image_file, model=self.model)
                system_prompt = ENGLISH_QWQ_PROMPT
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        return system_prompt


    def _generate_tool_calls(self, messages: list[dict], thinking: str) -> list[dict]:
        if self.visit_type == "summary":
            messages = [{"role": "system", "content": ENGLISH_SUMMARY_TOOL_SYSTEM_PROMPT}] + messages[1:] + [{"role": "assistant", "content": thinking}]
        elif self.visit_type == "content":
            messages = [{"role": "system", "content": ENGLISH_TOOL_SYSTEM_PROMPT}] + messages[1:] + [{"role": "assistant", "content": thinking}]
        else:
            raise ValueError(f"Unsupported visit_type: {self.visit_type}")
        try:
            tool_calls = call_openai(self.toolcall_model, messages, tools=self.tools, tool_choice="required")["tool_calls"]
        except Exception as e:
            tool_calls = []
        return tool_calls

    async def _act(self):
        # generate action step
        messages = self.memory.build_action_history()
        if not self.mock_thought and self.model == self.toolcall_model and self.model in REACT_MODEL:
            response = call_openai(self.model, messages, top_p=0.8, tools=self.tools, tool_choice="required")
            thinking = response["think"] + response["content"]
            tool_calls = response.get("tool_calls", [])
            if len(tool_calls) == 0:
                tool_calls = self.exit_tool_calls
        else:
            thinking = call_openai(self.model, messages)["content"]
            tool_calls = self._generate_tool_calls(messages, thinking)
            if len(tool_calls) == 0:
                tool_calls = self.exit_tool_calls
        action_step = ActionStep(thinking, tool_calls, self.model)
        self.memory.add_step(action_step)

        # execute action step
        done = False
        tool_steps = action_step.get_tool_call()
        for tool_step in tool_steps:
            tool_name = tool_step.get("name")
            tool_params = tool_step.get("arguments")
            if tool_name == "all_information_sufficient":
                done = True
                break
            # print(f"action_type: {tool_name}, action_arguments: {tool_params}")
            tool_response = await self.mcp_client.call_tool(tool_name, tool_params)
            observation = tool_response.content[0].text
            self.memory.add_step(ObservationStep(tool_name, observation))
        return done

    async def __call__(self, query: str, plan_memory: list=None) -> str:
        self.memory.clear()
        system_prompt = self._get_system_prompt()
        self.memory.add_step(SystemStep(system_prompt))
        self.memory.add_step(UserStep(query))
        if plan_memory:
            self.memory.add_step_list(plan_memory)
        step_count = 0
        done = False
        while step_count < self.max_steps and not done:
            done = await self._act()
            step_count += 1
        messages = self.memory.build_action_history()[1:]
        return done, messages
    
    async def cleanup(self):
        try:
            await self.exit_stack.aclose() 
        except Exception as e:
            pass

async def main():

    action_agent = ReactAgent(model="QwQ-32B", mock_thought=True)
    mcp_client = MCPClient()
    await mcp_client.connect_to_server("search_only")
    response = await mcp_client.list_tools()
    tools = [{
        "type": "function",
        "function": {
            "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
    action_agent.set_mcp(mcp_client, tools)
    observation = await mcp_client.call_tool("visit_urls", {"urls": ["https://es.wikipedia.org/wiki/Mambrilla_de_Castrej%C3%B3n"]})
    print(observation)
    import pdb; pdb.set_trace()
    try:
        done, action_history = await action_agent(query="The arena where the Lewiston Maineiacs played their home games can seat how many people?", plan_memory=None)
        print(done)
        print(len(action_history))
    finally:
        await action_agent.cleanup()
        await mcp_client.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
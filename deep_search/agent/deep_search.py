import asyncio
import os
import re
import json
from typing import List, Tuple
from contextlib import AsyncExitStack

from search_toolkit.environments.basic_environment import BasicSearchEnvironment

from deep_search.agent.response_agent import ResponseAgent
from deep_search.agent.plan_agent import PlanAgent, KeyPointsExtractor
from deep_search.agent.obs_evaluator import EvalObservation
from deep_search.agent.action_agent import ActionAgent
from deep_search.agent.react_agent import ReactAgent
from deep_search.agent.tot_agent import ToTAgent
from deep_search.utils.memory import StepManager, BaseStep, ActionStep, UserStep, SystemStep, NoticeStep, ToolStep, KeyPointsStep, MissingInfoNoticeStep, ObservationStep, TaskStep
from deep_search.utils.inspectors import get_zip_description, get_single_file_description, visualizer, TextInspectorTool
from deep_search.utils.retry import retry
from deep_search.utils.api import call_openai, gpt_json_load

from search_toolkit.mcp.client import MCPClient

class DeepSearch:

    def __init__(self, model: str, toolcall_model: str=None, task: str=None, attachments: dict={}, max_steps: int=12, simu_steps: int=12, agent_work_dir: str="./download", 
            obs_review_max_turn: int=1, mcp_server_name: str="simple", plan_strategy: str="fact", expand_num: int=2, evaluator_configs: dict=None,
            method_type: str="hierarchical", **kwargs):
        self.model = model
        self.method_type = method_type

        self.toolcall_model = toolcall_model if toolcall_model else model
        self.task = task
        self.attachments = attachments
        self.max_steps = max_steps
        self.simu_steps = simu_steps
        self.agent_work_dir = os.path.abspath(agent_work_dir)

        self.memory = StepManager(model=self.model, agent_work_dir=self.agent_work_dir)
        self.key_points_extractor = KeyPointsExtractor(model=self.model)
        self.obs_evaluator = EvalObservation(model=self.model)
        self.retry_count = 0
        self.max_retry = obs_review_max_turn

        self.plan_strategy = plan_strategy
        self.plan_agent = PlanAgent(model=self.model, agent_work_dir=self.agent_work_dir)
        
        self.mcp_server_name = mcp_server_name
        self.mcp_client = None
        self.tools = None
        if self.method_type == "hierarchical":
            sub_agent_type = kwargs.get("sub_agent", "monte-carlo")
            if sub_agent_type == "monte-carlo":
                self.action_agent = ActionAgent(model=self.model, toolcall_model=self.toolcall_model, agent_work_dir=self.agent_work_dir, expand_num=expand_num, max_steps=self.simu_steps, evaluator_configs=evaluator_configs, **kwargs)
            elif sub_agent_type == "tot":
                self.action_agent = ToTAgent(model=self.model, toolcall_model=self.toolcall_model, agent_work_dir=self.agent_work_dir, expand_num=expand_num, need_summary=True, **kwargs)
            else:
                raise ValueError(f"Invalid sub-agent type: {sub_agent_type}")
        elif self.method_type == "react":
            self.action_agent = ReactAgent(model=self.model, toolcall_model=self.toolcall_model, agent_work_dir=self.agent_work_dir, **kwargs)
        elif self.method_type == "tot":
            self.action_agent = ToTAgent(model=self.model, toolcall_model=self.toolcall_model, agent_work_dir=self.agent_work_dir, expand_num=expand_num, **kwargs)

        self.visualizer = visualizer
        self.text_inspector = TextInspectorTool(model=self.model, text_limit=1000)

        self.exit_stack = AsyncExitStack()
        self.search_message_type = kwargs.get("search_message_type", "raw")
        
        # Token usage statistics
        self.token_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0
        }

    def _update_token_usage(self, usage_info: dict):
        """Update token usage statistics"""
        if usage_info:
            self.token_usage["total_prompt_tokens"] += usage_info.get("prompt_tokens", 0)
            self.token_usage["total_completion_tokens"] += usage_info.get("completion_tokens", 0)
            self.token_usage["total_tokens"] += usage_info.get("total_tokens", 0)
            self.token_usage["api_calls"] += 1

    def get_token_usage(self) -> dict:
        """Get current token usage statistics"""
        return self.token_usage.copy()

    async def initialize(self):
        self.mcp_client = MCPClient()
        await self.mcp_client.connect_to_server(self.mcp_server_name)
        self.tools = await self._get_tools()
        self.action_agent.set_mcp(self.mcp_client, self.tools)

    async def _get_tools(self):
        response = await self.mcp_client.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        return available_tools

    def get_search_info(self) -> str:
        return self.memory.format_search_info()

    def _init_memory(self, query: str, attachments: dict=None) -> str:
        self.memory.clear()
        self.memory.add_step(self._build_user_memory(query=query, attachments=attachments))

    def _build_user_memory(self, query: str, attachments: dict=None) -> str:
        return UserStep(query, attachments, model=self.model)

    def _build_attachments(self, query: str, attachments: dict) -> dict:
        attachments_description = ""
        if attachments:
            # prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            for file_name, file_path in attachments.items():
                if ".zip" in file_name:
                    attachments_description += get_zip_description(
                        file_path, query, self.visualizer, self.text_inspector
                    )
                else:
                    attachments_description += get_single_file_description(
                        file_path, query, self.visualizer, self.text_inspector
                    )
        return attachments_description

    def _system_message(self) -> str:
        if self.plan_strategy != "none":
            system_prompt = """You are a professional intelligent search assistant, skilled at generating the next search task to execute based on the user's task requirements and historical messages. Users will break down the Task into multiple subtasks and complete the entire task by gradually completing these subtasks. Your task is to determine whether the current task is complete based on the user's initial question, and if not, generate the next search task to execute.

## Task Analysis
1. Carefully read the task requirements provided by the user.
2. Analyze in detail the steps completed and the information obtained in historical messages.
3. Evaluate the current task progress, identify missing information, and determine the next problem to solve.

## Task Status Assessment
Please determine whether the current task is complete:
- If all key information points have been obtained, please output:
{ "reasoning": "Explain in detail why the task is considered complete", "task_completed": true, "next_task": null }
- If the task is not yet complete, please output:
{
    "reasoning": "Explain in detail why the task is not considered complete", 
    "task_completed": false, 
    "next_task": {
        "reasoning": "Why this is the most appropriate next task",
        "next_task": "Specifically describe the next task to execute",
        "expected_outcome": "Key information points expected to be obtained by executing this task"
    }
}

## Notes
- Ensure each generated task is clear, specific, and executable.
- Ensure each task focuses on only one information point.
- Avoid repeating steps that have already been completed.
- Prioritize solving the most critical information for completing the overall task.
- Consider dependencies between tasks to ensure logical order.
- Please output in JSON format only, do not output any other content.
"""
        else: 
            system_prompt = """You are a professional intelligent search assistant, skilled at generating the next search task to execute based on the user's task requirements and historical messages. Users will break down the Task into multiple subtasks and complete the entire task by gradually completing these subtasks. You should determine the next search task to execute based on the user's initial question and historical messages.

## Task Analysis
1. Carefully read the task requirements provided by the user.
2. Analyze in detail the steps completed and the information obtained in historical messages.
3. Evaluate the current task progress, identify missing information, and determine the next problem to solve.
"""
        return {
            "role": "system",
            "content": system_prompt
        }
    
    def _init_system_message(self) -> str:
        system_prompt = """You are a professional intelligent search assistant, skilled at generating the next search task to execute based on the user's task requirements and historical messages. Users will break down the Task into multiple subtasks and complete the entire task by gradually completing these subtasks. You should determine the next search task to execute based on the user's initial question and historical messages.

## Task Analysis
1. Carefully read the task requirements provided by the user.
2. Analyze in detail the steps completed and the information obtained in historical messages.
3. Evaluate the current task progress, identify missing information, and determine the next problem to solve.

## Output Format
You should output the next search task to execute in the following format, task_completed is always false:
  {
    "reasoning": "Explain in detail why the task is not considered complete", 
    "task_completed": false, 
    "next_task": {
        "reasoning": "Why this is the most appropriate next task",
        "next_task": "Specifically describe the next task to execute",
        "expected_outcome": "Key information points expected to be obtained by executing this task"
    }
  }

## Notes
- Ensure each generated task is clear, specific, and executable.
- Ensure each task focuses on only one information point.
- Avoid repeating steps that have already been completed.
- Prioritize solving the most critical information for completing the overall task.
- Consider dependencies between tasks to ensure logical order.
- Please output in JSON format only, do not output any other content.
"""
        return {
            "role": "system",
            "content": system_prompt
        }

    def _user_message(self, query: str, key_points: str) -> str:
        user_prompt = f"""## Task
{query}

## Key Points
{key_points}
"""
        return {
            "role": "user",
            "content": user_prompt
        }
    
    def _build_messages(self) -> list[dict]:
        action_history = self.memory.build_action_history()
        if (self.plan_strategy == "planning" and len(action_history) == 2) or (self.plan_strategy in ["none", "plan_only", "fact"] and len(action_history) == 1) or (self.plan_strategy == "none" and len(action_history) == 0):
            return [self._init_system_message()] + action_history
        else:
            return [self._system_message()] + action_history

    @retry(3, 1)
    def _generate_next_task(self) -> List[str]:
        messages = self._build_messages()
        response = call_openai(self.model, messages)
        # Update token usage
        self._update_token_usage(response.get("usage", {}))
        next_task_info = gpt_json_load(response['content'])
        return next_task_info['next_task'], next_task_info['task_completed']
    
    def _merge_sub_tasks(self, sub_task_records: List[dict]):
        # TODO: 需要完善sub task merging
        pass

    @retry(10, 1)
    async def _act(self, query: str, history_messages: list[dict]):
        next_task_info, done = self._generate_next_task()
        if not done:
            next_task = next_task_info['next_task']
            print("-"*10, f"next_task: {next_task}", "-"*10, sep="\n")
            trajectory, summary, search_messages, other_info = await self.action_agent(query=next_task, history_messages=history_messages)
            # Collect token usage from action agent
            if hasattr(self.action_agent, 'get_token_usage'):
                action_agent_usage = self.action_agent.get_token_usage()
                # Merge token usage statistics
                self.token_usage["total_prompt_tokens"] += action_agent_usage.get("total_prompt_tokens", 0)
                self.token_usage["total_completion_tokens"] += action_agent_usage.get("total_completion_tokens", 0)
                self.token_usage["total_tokens"] += action_agent_usage.get("total_tokens", 0)
                self.token_usage["api_calls"] += action_agent_usage.get("api_calls", 0)
            other_info["next_task"] = next_task
            other_info["summary"] = summary
            self.memory.add_step(TaskStep(next_task, summary, trajectory, other_info))
        else:
            search_messages = []
            other_info = {"next_task": None, "summary": None}
        return done, search_messages, other_info

    async def _hierarchical_search(self) -> None:
        current_step = 0
        search_messages = []
        other_infos = []
        while current_step < self.max_steps:
            history_messages = self.memory.build_action_history()
            done, search_message, other_info = await self._act(self.task, history_messages)
            search_messages.extend(search_message)
            other_infos.append(other_info)
            if done:
                break
            current_step += 1
        # Add token usage to the last other_info
        if other_infos:
            other_infos[-1]["token_usage"] = self.get_token_usage()
        
        if self.search_message_type == "raw":
            return search_messages, other_infos
        elif self.search_message_type == "summary":
            return history_messages, other_infos
        else:
            raise ValueError(f"Invalid search message type: {self.search_message_type}")
    
    async def _react_search(self, plan_memory: str=None) -> None:
        done, search_messages = await self.action_agent(query=self.task, plan_memory=plan_memory)
        # Collect token usage from action agent
        if hasattr(self.action_agent, 'get_token_usage'):
            action_agent_usage = self.action_agent.get_token_usage()
            # Merge token usage statistics
            self.token_usage["total_prompt_tokens"] += action_agent_usage.get("total_prompt_tokens", 0)
            self.token_usage["total_completion_tokens"] += action_agent_usage.get("total_completion_tokens", 0)
            self.token_usage["total_tokens"] += action_agent_usage.get("total_tokens", 0)
            self.token_usage["api_calls"] += action_agent_usage.get("api_calls", 0)
        return search_messages
    
    async def _tot_search(self, plan_memory: str=None) -> None:
        trajectory, summary, action_history, other_info = await self.action_agent(query=self.task, fact=plan_memory)
        # Collect token usage from action agent
        if hasattr(self.action_agent, 'get_token_usage'):
            action_agent_usage = self.action_agent.get_token_usage()
            # Merge token usage statistics
            self.token_usage["total_prompt_tokens"] += action_agent_usage.get("total_prompt_tokens", 0)
            self.token_usage["total_completion_tokens"] += action_agent_usage.get("total_completion_tokens", 0)
            self.token_usage["total_tokens"] += action_agent_usage.get("total_tokens", 0)
            self.token_usage["api_calls"] += action_agent_usage.get("api_calls", 0)
        # Add token usage to other_info
        other_info["token_usage"] = self.get_token_usage()
        return trajectory, other_info

    async def __call__(self) -> None:
        attachments_description = self._build_attachments(self.task, self.attachments)
        if self.plan_strategy == "fact":
            plan_memory = [self.plan_agent.fact_extraction(self.task, attachments_description=attachments_description)]
            # Collect token usage from plan agent
            if hasattr(self.plan_agent, 'get_token_usage'):
                plan_agent_usage = self.plan_agent.get_token_usage()
                self.token_usage["total_prompt_tokens"] += plan_agent_usage.get("total_prompt_tokens", 0)
                self.token_usage["total_completion_tokens"] += plan_agent_usage.get("total_completion_tokens", 0)
                self.token_usage["total_tokens"] += plan_agent_usage.get("total_tokens", 0)
                self.token_usage["api_calls"] += plan_agent_usage.get("api_calls", 0)
        elif self.plan_strategy == "planning":
            plan_memory = self.plan_agent.plan(self.task, attachments_description=attachments_description)
            # Collect token usage from plan agent
            if hasattr(self.plan_agent, 'get_token_usage'):
                plan_agent_usage = self.plan_agent.get_token_usage()
                self.token_usage["total_prompt_tokens"] += plan_agent_usage.get("total_prompt_tokens", 0)
                self.token_usage["total_completion_tokens"] += plan_agent_usage.get("total_completion_tokens", 0)
                self.token_usage["total_tokens"] += plan_agent_usage.get("total_tokens", 0)
                self.token_usage["api_calls"] += plan_agent_usage.get("api_calls", 0)
        elif self.plan_strategy == "plan_only":
            plan_memory = self.plan_agent.plan(self.task, attachments_description=attachments_description)
            # Collect token usage from plan agent
            if hasattr(self.plan_agent, 'get_token_usage'):
                plan_agent_usage = self.plan_agent.get_token_usage()
                self.token_usage["total_prompt_tokens"] += plan_agent_usage.get("total_prompt_tokens", 0)
                self.token_usage["total_completion_tokens"] += plan_agent_usage.get("total_completion_tokens", 0)
                self.token_usage["total_tokens"] += plan_agent_usage.get("total_tokens", 0)
                self.token_usage["api_calls"] += plan_agent_usage.get("api_calls", 0)
            plan_memory = [plan_memory[-1]]
        elif self.plan_strategy == "none":
            plan_memory = None
        else:
            raise ValueError(f"Invalid plan strategy: {self.plan_strategy}")
        self._init_memory(self.task, attachments_description)
        if self.method_type == "hierarchical":
            search_messages, other_info = await self._hierarchical_search()
            return [{"role": "user", "content": self.task}] + search_messages, other_info
        elif self.method_type == "react":
            return await self._react_search(plan_memory), None
        elif self.method_type == "tot":
            trajectory, other_info = await self._tot_search(plan_memory)
            return trajectory, other_info
        else:
            raise ValueError(f"Invalid method type: {self.method_type}")
    
    async def cleanup(self):
        try:
            await self.mcp_client.cleanup()
        finally:
            pass

async def main():
    # query = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    query = "Were Scott Derrickson and Ed Wood of the same nationality?"
    try:
        deep_search = DeepSearch(model="QwQ-32B", task=query, obs_review_max_turn=0, mcp_server_name="search_only", method_type="react")
        await deep_search.initialize()
        search_messages, other_info = await deep_search()
        response_agent = ResponseAgent(model="QwQ-32B")
        answer = response_agent.response(query=query, messages=search_messages)
        print(answer)
    finally:
        await deep_search.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

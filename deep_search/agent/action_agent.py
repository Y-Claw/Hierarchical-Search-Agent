import os
import json
import math
import random
from contextlib import AsyncExitStack
from functools import reduce
import numpy as np

from deep_search.agent.prompts.agent_prompting import get_full_system_prompt, ENGLISH_OPENAI_PROMPT, ENGLISH_TOOL_SYSTEM_PROMPT, ENGLISH_QWQ_PROMPT
from deep_search.agent.state_evaluator.information_evaluator import InformationEvaluator
from deep_search.utils.api import call_openai
from deep_search.utils.memory import ActionStep, ToolStep, BaseStep, StepManager, ObservationStep, NoticeStep
from deep_search.utils.agent_utils import get_have_internet, current_datetime
from deep_search.utils.enums import QWEN_MODEL, QWQ_MODEL, OPENAI_MODEL
from deep_search.utils.config import DOWNLOAD_DIR

from search_toolkit.mcp.client import MCPClient

# New MCTS related classes
class MCTSNode:
    def __init__(self, state, parent=None, model="QwQ-32B"):
        self.state = state  # Current state (history, context, etc.)
        self.model = model
        self.parent = parent
        self.children = []  # List of (action, child_node) tuples
        self.possible_actions = []
        self.visits = 0
        self.value = 0.0
        self.terminal = False
        
    def is_fully_expanded(self):
        return all([child[1].is_terminal() for child in self.children])
    
    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None
            
        # UCB1 formula for selection
        def ucb_score(child_node):
            if child_node.is_terminal():
                return float('-inf')
            exploitation = child_node.value / child_node.visits if child_node.visits > 0 else 0
            try:
                exploration = exploration_weight * math.sqrt(2 * math.log(self.visits) / child_node.visits) if child_node.visits > 0 else float('inf')
            except Exception as e:
                import pdb; pdb.set_trace()
            return exploitation + exploration

        return max(self.children, key=lambda child: ucb_score(child[1]))[1]

    def is_terminal(self):
        if self.parent is None:
            return self.terminal
        return any([tool_call["name"] == "all_information_sufficient" for tool_call in self.state["action"]]) or self.terminal

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, thinking, child_node):
        self.children.append((thinking, child_node))

    def calculate_prob(self):
        """Calculate probability for this child node based on visits"""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def _get_sys_message(self) -> list[dict]:
        if self.model in OPENAI_MODEL:
            system_message = ENGLISH_OPENAI_PROMPT
        elif self.model in QWEN_MODEL or self.model in QWQ_MODEL:
            # system_message = get_full_system_prompt(agent_work_dir=DOWNLOAD_DIR, text_context_list=self.state["text_context"], image_file=self.state["image_file"], model=self.model)
            system_message = ENGLISH_QWQ_PROMPT
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        return [{"role": "system", "content": system_message}]

    def _get_user_message(self) -> list[dict]:
        return [{"role": "user", "content": self.state["query"]}]

    def _get_action_message(self) -> list[dict]:
        action_str = "\n".join([f"Tool: {tool_call['name']}\n Arguments: {tool_call['arguments']}" for tool_call in self.state["action"]])
        return [{"role": "assistant", "content": self.state["thinking"]}, {"role": self.state["action_role"], "content": action_str}, {"role": "user", "content": self.state["observation"]}]

    def get_action_message(self) -> str:
        if self.parent is None:
            system_message = self._get_sys_message()
            user_message = self._get_user_message()
            return system_message + user_message
        else:
            history_message = self.parent.get_action_message()
            return history_message + self._get_action_message()

    def backpropagate(self, reward: float):
        node = self
        while node.parent:
            node.visits += 1
            node.value = node.value + (reward - node.value) / node.visits
            node.terminal = node.is_fully_expanded()
            node = node.parent
        node.visits += 1
        node.value = node.value + (reward - node.value) / node.visits
        node.terminal = node.is_fully_expanded()

    def get_action_history(self):
        action_history = []
        node = self.best_child(exploration_weight=0.0)
        while not node.is_leaf():
            action_history.extend(node._get_action_message())
            node = node.best_child()
        return action_history, node

    def print_tree(self, indent=0, prefix=""):
        # Print current node information
        node_info = f"{prefix}[Visits: {self.visits}, Value: {self.value:.4f}]"
        
        # Print action information if available
        if "action" in self.state and self.state["action"]:
            action_names = [tool_call["name"] for tool_call in self.state["action"]]
            node_info += f" Actions: {', '.join(action_names)}"
        
        print(" " * indent + node_info)
        
        # Print children with increased indentation
        for i, (thinking, child) in enumerate(self.children):
            is_last = i == len(self.children) - 1
            child_prefix = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "
            
            # Print a shortened version of thinking
            thinking_short = thinking[:50] + "..." if thinking and len(thinking) > 50 else thinking
            print(" " * indent + child_prefix + f"Thinking: {thinking_short}")
            
            # Recursively print child nodes
            child.print_tree(indent + 4, prefix=next_prefix)
    
    def get_tree(self):
        def build_tree(node):
            node_info = {
                "visits": node.visits,
                "value": node.value,
                "actions": [tool_call["name"] for tool_call in node.state["action"]] if "action" in node.state and node.state["action"] else []
            }
            children_info = []
            for thinking, child in node.children:
                child_info = {
                    "thinking": thinking,
                    "child": build_tree(child)
                }
                children_info.append(child_info)
            node_info["children"] = children_info
            return node_info

        return build_tree(self)

class ActionAgent:
    
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
        
        # MCTS related parameters
        self.expand_num = expand_num
        self.mcts_exploration_weight = mcts_exploration_weight
        self.mcts_root = None  # Global root node for MCTS
        self.state_accumulate_type = kwargs.get("state_accumulate_type", "add")
        self.evaluators = self._build_evaluators(evaluator_configs)

    def _build_evaluators(self, evaluator_configs: dict):
        evaluators = []
        for evaluator_name, evaluator_config in evaluator_configs.items():
            if evaluator_name == "information_evaluator":
                evaluators.append(InformationEvaluator(self.model, **evaluator_config))
        return evaluators

    def set_mcp(self, mcp_client: MCPClient, tools: list[dict]):
        self.mcp_client = mcp_client
        self.tools = tools
    
    def _select(self):
        # Implementation will depend on your specific needs
        # This is a simplified version
        current = self.mcts_root
        while not current.is_leaf():
            child = current.best_child()
            current = child
        
        return current
    
    def _expand(self, current):
        possible_action_steps = self._generate_tool_steps(current)
        
        for action_step in possible_action_steps:
            state = {
                "thinking": action_step.get_thinking(),
                "action": action_step.get_tool_call(),
                "model": self.model,
                "action_role": "assistant",
                "observation": "",
            }
            child = MCTSNode(state, parent=current)
            current.add_child(action_step.get_thinking(), child)

        return current
    
    def _evaluate_state(self, query: str, state: dict):
        # Evaluate how good the state is
        # This could be based on various heuristics
        # Simplified example - you'd need a more sophisticated evaluation
        # TODO: metrics
        #   1. 当前action是否获取到与用户query相关的信息
        #   2. 当前action的thinking是否合理
        #   3. tool_call 是否与 thinking 一致
        # Here are more metrics from gpt"
        #   1. 信息新鲜度 (Freshness): 判断获取到的信息是否最新，尤其对时效性强的问题至关重要。
        #   2. 信息来源可信度 (Source Credibility): 评估数据来源的权威性、专业性、可靠性，防止被误导。
        #   3. 多样性与覆盖度 (Diversity & Coverage): 判断当前获取的信息是否多元，是否覆盖到 query 涉及的主要方面，避免信息片面。
        #   4. 交互代价 (Interaction Cost): 评估当前action是否值得继续探索，考虑API调用成本、响应时间、查询轮次等因素。
        #   5. 上下文一致性 (Context Consistency): 检查当前状态是否与之前的思路、已获取信息保持逻辑连贯，避免跑偏。
        #   6. 用户意图匹配度 (User Intent Alignment): 不仅匹配 query 字面意思，还要考虑 query 背后的深层意图，确保方向不偏。
        #   7. 探索与利用平衡 (Exploration-Exploitation Trade-off): 判断是否应该继续探索新路径，还是利用已有信息加速收敛，平衡好探索和利用。
        #   8. 信息置信度 (Confidence Score): 为当前获取的信息打分，结合大模型推理能力判断信息是否"足够好"或"需要补充"。
        if self.state_accumulate_type == "add":
            return sum([evaluator.evaluate(query, state) for evaluator in self.evaluators]) / len(self.evaluators)
        elif self.state_accumulate_type == "multiply":
            return reduce(lambda x, y: x * y, [evaluator.evaluate(query, state) for evaluator in self.evaluators])
        else:
            raise ValueError(f"Invalid state accumulate type: {self.state_accumulate_type}")

    def _generate_tool_calls(self, messages: list[dict], thinking: str) -> list[dict]:
        messages = [{"role": "system", "content": ENGLISH_TOOL_SYSTEM_PROMPT}] + messages[1:] + [{"role": "user", "content": thinking}]
        try:
            tool_calls = call_openai(self.toolcall_model, messages, tools=self.tools, tool_choice="required")["tool_calls"]
        except Exception as e:
            tool_calls = []
        return tool_calls

    def _generate_tool_steps(self, current) -> str:
        # Original implementation
        message = current.get_action_message()
        action_steps = []
        for _ in range(self.expand_num):
            if self.model == self.toolcall_model and (self.model in QWEN_MODEL or self.model in QWQ_MODEL):
                response = call_openai(self.model, message, top_p=0.8, tools=self.tools, tool_choice="required")
                thinking = response["think"] + response["content"]
                tool_calls = response.get("tool_calls", [])
                if len(tool_calls) == 0:
                    tool_calls = self.exit_tool_calls
            else:
                thinking = call_openai(self.model, message)["content"]
                tool_calls = self._generate_tool_calls(message, thinking)
            action_step = ActionStep(thinking, tool_calls, self.model)
            action_steps.append(action_step)
        return action_steps

    async def _mcts_step(self, query: str, history_messages: list[dict], text_context_list: list[str]=None, image_file: str=None):
        # Selection
        current_step = self._select()
        # Simulation
        observation, done = await self._execute_action(current_step.state["action"])
        current_step.state["observation"] = observation
        reward = self._evaluate_state(query, current_step.state)

        # Expansion
        current_step = self._expand(current_step)

        # Backpropagation
        current_step.backpropagate(reward)

    async def _execute_action(self, tool_steps: list[dict]):
        observations = []
        done = False
        for tool_step in tool_steps:
            tool_name = tool_step.get("name")
            tool_params = tool_step.get("arguments")
            # print(f"action_type: {tool_name}, action_arguments: {tool_params}")
            # self.memory.add_step(tool_step)
            if tool_name == "all_information_sufficient":
                done = True
                break
            observation = await self.mcp_client.call_tool(tool_name, tool_params)
            observations.append(observation)
        observation = "\n".join([observations[i].content[0].text for i in range(len(observations))])
        return observation, done

    def _get_trajectory(self):
        action_history, final_node = self.mcts_root.get_action_history()
        trajectory = ""
        for action in action_history:
            trajectory += action["content"] + "\n"
        return trajectory, action_history

    def _summarize_trajectory(self, query: str, trajectory: str):
        # TODO: 需要完善trajectory summary
        system_prompt = """You are a professional intelligent search assistant, summarizing the execution of the current task. The current task is a subtask of a complete task.

## Task Requirements
1. Summarize the execution of the current subtask, including the thought process described, the tools used, and the results of the tools' execution.
2. Determine whether the current subtask is completed.
3. If the subtask is not completed, identify which step in the current execution caused the subtask to be incomplete.
4. If the subtask is completed, provide a summary.
5. Descriptions of search-related tool calls need to include the full web link.

"""
        user_prompt = f"""Current subtask: {query}
Execution record: {trajectory}"""
        summary = call_openai(self.model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])["content"]
        return summary

    async def __call__(self, query: str, history_messages: list[dict], text_context_list: list[str]=None, image_file: str=None) -> str:
        step_count = 0
        self.memory.clear()
        # Reset MCTS root when starting a new query
        self.mcts_root = MCTSNode(model=self.model,
            state={
            "query": query,
            "history": history_messages,
            "text_context": text_context_list,
            "image_file": image_file,
            "action_history": self.memory.build_action_history(),
        })

        self._expand(self.mcts_root)
        while step_count < self.max_steps and not self.mcts_root.is_terminal():
            await self._mcts_step(query, self.mcts_root, text_context_list, image_file)
            step_count += 1
        trajectory, action_history = self._get_trajectory()
        summary = self._summarize_trajectory(query, trajectory)
        return trajectory, summary, action_history, {"tree": self.mcts_root.get_tree()}
    
    async def cleanup(self):
        try:
            await self.exit_stack.aclose() 
        except Exception as e:
            pass

async def main():
    # action_agent = ActionAgent(think_model="models/QwQ-32B", toolcall_model="models/QwQ-32B")
    evaluator_configs = {
        "information_evaluator": {
            "score_map": {
                1: 2,
                2: 1,
                3: 0    
            }
        }
    }
    action_agent = ActionAgent(model="QwQ-32B", evaluator_configs=evaluator_configs)
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
    try:
        trajectory, summary, action_history, search_list = await action_agent(query="The arena where the Lewiston Maineiacs played their home games can seat how many people?", history_messages=[], text_context_list=[], image_file=None)
        print(trajectory)
        print(summary)
        print(len(action_history))
        action_agent.mcts_root.print_tree()
    finally:
        await action_agent.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

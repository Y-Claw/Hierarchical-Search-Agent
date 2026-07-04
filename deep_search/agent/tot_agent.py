import os
import json
import math
import numpy as np
import random
from contextlib import AsyncExitStack
from functools import reduce
import concurrent.futures
import asyncio
import time

from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

from deep_search.agent.prompts.agent_prompting import get_full_system_prompt, OPENAI_PROMPT, TOOL_SYSTEM_PROMPT, MOCK_QWQ_PROMPT, ENGLISH_OPENAI_PROMPT, ENGLISH_TOOL_SYSTEM_PROMPT, ENGLISH_SUMMARY_TOOL_SYSTEM_PROMPT, ENGLISH_QWQ_PROMPT, ENGLISH_SUMMARY_QWQ_PROMPT, ENGLISH_QWQ_VISIT_ONLY_PROMPT, ENGLISH_QWQ_ASK_ONLY_PROMPT
from deep_search.agent.state_evaluator import EVALUATORS
from deep_search.agent.response_agent import ResponseAgent
from deep_search.utils.api import call_openai
from deep_search.utils.memory import ActionStep, ToolStep, BaseStep, StepManager, ObservationStep, NoticeStep, UserStep, SystemStep
from deep_search.utils.agent_utils import get_have_internet, current_datetime
from deep_search.utils.enums import QWEN_MODEL, QWQ_MODEL, OPENAI_MODEL, REACT_MODEL, FUNC_ONLY_MODEL
from deep_search.utils.config import DOWNLOAD_DIR

from search_toolkit.mcp.client import MCPClient

# New MCTS related classes
class ToTNode:
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
        if self.state["fact"]:
            return [{"role": "user", "content": self.state["query"]}, {"role": "assistant", "content": self.state["fact"]}]
        else:
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

class ToTAgent:
    
    def __init__(self, model: str, toolcall_model: str=None, agent_work_dir: str=None, mcp_server_name: str=None, max_steps: int=16, 
                 expand_num: int=2, n_select_sample: int=2, state_evaluator_configs: dict=None, process_evaluator_configs: dict=None, need_summary: bool=False, **kwargs):
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
        self.response_agent = ResponseAgent(model=self.model)
        self.exit_stack = AsyncExitStack()
        self.exit_tool_calls = [{"type": "function", "function": {"name": "all_information_sufficient", "arguments": {}}}]
        
        # MCTS related parameters
        self.expand_num = expand_num
        self.n_select_sample = n_select_sample
        self.tot_root = None  # Global root node for MCTS
        self.state_accumulate_type = kwargs.get("state_accumulate_type", "add")
        self.state_evaluators = self._build_evaluators(state_evaluator_configs)
        self.process_accumulate_type = kwargs.get("process_accumulate_type", "add")
        self.process_evaluators = self._build_evaluators(process_evaluator_configs)
        self.retain_last_observation = kwargs.get("retain_last_observation", False)
        self.no_select_finish = kwargs.get("no_select_finish", False)

        self.need_summary = need_summary
        self.visit_type = kwargs.get("visit_type", "content")
        
        # Token usage statistics
        self.token_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0
        }

    def _build_evaluators(self, evaluator_configs: dict):
        evaluators = []
        for evaluator_name, evaluator_config in evaluator_configs.items():
            evaluators.append(EVALUATORS[evaluator_name](self.model, **evaluator_config))
        return evaluators

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

    def set_mcp(self, mcp_client: MCPClient, tools: list[dict]):
        self.mcp_client = mcp_client
        self.tools = tools
    
    def _select(self, current_steps: list[ToTNode], rewards: list[float], select_ids: list[int]):
        # Implementation will depend on your specific needs
        # This is a simplified version
        # ids = list(range(len(current_steps)))
        ids = select_ids
        select_ids = sorted(ids, key=lambda x: rewards[x], reverse=True)[:self.n_select_sample]
        select_new_steps = [current_steps[select_id] for select_id in select_ids]
        return select_new_steps
    
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
            return sum([evaluator.evaluate(query, state) for evaluator in self.state_evaluators]) / len(self.state_evaluators)
        elif self.state_accumulate_type == "multiply":
            return reduce(lambda x, y: x * y, [evaluator.evaluate(query, state) for evaluator in self.state_evaluators])
        else:
            raise ValueError(f"Invalid state accumulate type: {self.state_accumulate_type}")

    def _evaluate_process(self, query: str, messages: list[dict]):
        return sum([evaluator.evaluate(query, messages) for evaluator in self.process_evaluators]) / len(self.process_evaluators)
    
    def _evaluate_process_all(self, query: str, messages_list: list[dict]):
        # 根据累积类型初始化scores_result
        if self.process_accumulate_type == "add":
            scores_result = [0] * len(messages_list)
        elif self.process_accumulate_type == "multiply":
            scores_result = [1] * len(messages_list)  # multiply时初值应为1
        else:
            raise ValueError(f"Invalid process accumulate type: {self.process_accumulate_type}")
        
        # 并发执行所有evaluator的evaluate_all方法
        def evaluate_single(evaluator):
            return evaluator.evaluate_all(query, messages_list)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交所有任务
            future_to_evaluator = {executor.submit(evaluate_single, evaluator): evaluator 
                                  for evaluator in self.process_evaluators}
            
            # 收集结果并累积
            for future in concurrent.futures.as_completed(future_to_evaluator):
                scores = future.result()
                if self.process_accumulate_type == "add":
                    scores_result = [(scores_result[i] + scores[i]) for i in range(len(scores))]
                elif self.process_accumulate_type == "multiply":
                    scores_result = [(scores_result[i] * scores[i]) for i in range(len(scores))]
        
        return scores_result

    def _generate_tool_calls(self, messages: list[dict], thinking: str) -> list[dict]:
        if self.visit_type == "content":
            messages = [{"role": "system", "content": ENGLISH_TOOL_SYSTEM_PROMPT}] + messages[1:] + [{"role": "user", "content": thinking}]
        elif self.visit_type == "summary":
            messages = [{"role": "system", "content": ENGLISH_TOOL_SYSTEM_PROMPT}] + messages[1:] + [{"role": "user", "content": thinking}]
        else:
            raise ValueError(f"Invalid visit type: {self.visit_type}")
        try:
            response = call_openai(self.toolcall_model, messages, tools=self.tools, tool_choice="required")
            # Update token usage
            self._update_token_usage(response.get("usage", {}))
            tool_calls = response["tool_calls"]
        except Exception as e:
            tool_calls = []
        return tool_calls

    def _generate_single_tool_step(self, current) -> ActionStep:
        """Generate a single action step for a given current node"""
        message = current.get_action_message()
        
        if self.model == self.toolcall_model and (self.model in QWEN_MODEL or self.model in QWQ_MODEL):
            if self.mcp_server_name == "visit_only":
                message[0]["content"] = ENGLISH_QWQ_VISIT_ONLY_PROMPT
            elif self.mcp_server_name == "ask_only":
                message[0]["content"] = ENGLISH_QWQ_ASK_ONLY_PROMPT
            response = call_openai(self.model, message, top_p=0.8, tools=self.tools, tool_choice="required")
            # Update token usage
            self._update_token_usage(response.get("usage", {}))
            thinking = response["think"] + response["content"]
            tool_calls = response.get("tool_calls", [])
            if len(tool_calls) == 0:
                tool_calls = self.exit_tool_calls
        else:
            response = call_openai(self.model, message)
            # Update token usage
            self._update_token_usage(response.get("usage", {}))
            thinking = response["content"]
            tool_calls = self._generate_tool_calls(message, thinking)
        return ActionStep(thinking, tool_calls, self.model)

    def _create_child_node(self, action_step: ActionStep, parent: ToTNode) -> ToTNode:
        """Create a child node from an action step"""
        state = {
            "thinking": action_step.get_thinking(),
            "action": action_step.get_tool_call(),
            "model": self.model,
            "action_role": "assistant",
            "observation": "",
        }
        child = ToTNode(state, parent=parent)
        parent.add_child(action_step.get_thinking(), child)
        return child

    async def _tot_step(self, query: str, history_messages: list[dict], text_context_list: list[str]=None, image_file: str=None):
        # Selection
        finish_steps = []
        current_steps = []
        for step in self.current_steps:
            if step.is_terminal():
                finish_steps.append(step)
            else:
                current_steps.append(step)
        time_start = time.time()
        
        # Expansion - 并发生成所有action steps
        follow_steps = []
        
        # 为每个current_step和每个expand_num创建任务
        tasks = []
        for current_step in current_steps:
            for _ in range(self.expand_num):
                tasks.append((current_step, self._generate_single_tool_step))
        
        # 并发执行所有action step生成
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_task = {executor.submit(task_func, current_step): (current_step, task_func) 
                             for current_step, task_func in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                current_step, _ = future_to_task[future]
                action_step = future.result()
                child_node = self._create_child_node(action_step, current_step)
                follow_steps.append(child_node)
        
        time_end = time.time()
        print(f"Time taken for expansion: {time_end - time_start} seconds")
        time_start = time.time()
        
        # Simulation
        rewards = []
        finish_idxs = []
        unfinish_idxs = []
        
        # First loop: execute actions concurrently
        done_status = {}
        for i, follow_step in enumerate(follow_steps):
            observation, done = await self._execute_action(follow_step.state["action"], thinking=follow_step.state["thinking"])
            follow_step.state["observation"] = observation
            done_status[i] = done
            
        # Second loop: evaluate states concurrently
        def evaluate_single_state(index_and_step):
            i, follow_step = index_and_step
            reward = self._evaluate_state(query, follow_step.state)
            return i, reward, done_status[i]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(evaluate_single_state, (i, follow_step)): i 
                             for i, follow_step in enumerate(follow_steps)}
            
            for future in concurrent.futures.as_completed(future_to_index):
                i, reward, is_done = future.result()
                rewards.append((i, reward))
                if is_done:
                    finish_idxs.append(i)
                else:
                    unfinish_idxs.append(i)
        
        # Sort rewards by original index to maintain order
        rewards.sort(key=lambda x: x[0])
        rewards = [reward for _, reward in rewards]
        
        time_end = time.time()
        print(f"Time taken for simulation: {time_end - time_start} seconds")
        
        time_start = time.time()
        # Select follow steps
        if self.no_select_finish:
            selected_follow_steps = self._select(follow_steps, rewards, unfinish_idxs)
            self.current_steps = finish_steps + [follow_steps[i] for i in finish_idxs] + selected_follow_steps
        else:
            selected_follow_steps = self._select(follow_steps, rewards, list(range(len(follow_steps))))
            self.current_steps = finish_steps + selected_follow_steps
        time_end = time.time()
        print(f"Time taken for select: {time_end - time_start} seconds")

    async def _execute_action(self, tool_steps: list[dict], thinking: str=""):
        observations = []
        final_observation = ""
        done = False
        for tool_step in tool_steps:
            tool_name = tool_step.get("name")
            tool_params = tool_step.get("arguments")
            if isinstance(tool_params, str):
                tool_params = json.loads(tool_params)
            # print(f"action_type: {tool_name}, action_arguments: {tool_params}")
            # self.memory.add_step(tool_step)
            if tool_name == "all_information_sufficient":
                done = True
                final_observation = thinking
                break
            try:
                observation = await self.mcp_client.call_tool(tool_name, tool_params)
            except Exception as e:
                pass
                # import pdb; pdb.set_trace()
            observations.append(observation)
        if self.retain_last_observation:
            observation = "\n".join([observations[i].content[0].text for i in range(len(observations))] + [final_observation])
        else:
            observation = "\n".join([observations[i].content[0].text for i in range(len(observations))])
        return observation, done

    def _get_trajectory(self, query: str):
        scores = []
        all_trajectory = []
        for node in self.current_steps:
            action_messages = node.get_action_message()[1:]
            all_trajectory.append(action_messages)
        scores = self._evaluate_process_all(query, all_trajectory)
        print(scores)
        selected_node = self.current_steps[np.argmax(scores)]
        return selected_node.get_action_message()[1:], all_trajectory, scores

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
        response = call_openai(self.model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        # Update token usage
        self._update_token_usage(response.get("usage", {}))
        summary = response["content"]
        return summary

    async def __call__(self, query: str, history_messages: list[dict] = None, text_context_list: list[str]=None, image_file: str=None, fact: str=None) -> str:
        step_count = 0
        self.memory.clear()
        # Reset Tot root when starting a new query
        self.tot_root = ToTNode(state={
            "query": query,
            "history": history_messages,
            "text_context": text_context_list,
            "image_file": image_file,
            "action_history": self.memory.build_action_history(),
            "fact": fact
        }, model=self.model)

        self.current_steps = [self.tot_root]
        while step_count < self.max_steps and any([not step.is_terminal() for step in self.current_steps]):
            start_time = time.time()
            await self._tot_step(query, self.tot_root, text_context_list, image_file)
            end_time = time.time()
            print(f"Time taken for step {step_count}: {end_time - start_time} seconds")
            step_count += 1
        trajectory, all_trajectory, scores = self._get_trajectory(query)
        if self.need_summary:
            summary = self._summarize_trajectory(query, trajectory)
        else:
            summary = ""
        other_info = {
            "all_trajectory": all_trajectory, 
            "scores": scores,
            "token_usage": self.get_token_usage()
        }
        return trajectory, summary, trajectory[1:], other_info
    
    async def cleanup(self):
        try:
            await self.exit_stack.aclose() 
        except Exception as e:
            pass

async def main():
    state_evaluator_configs = {
        "information_evaluator": {},
    }
    process_evaluator_configs = {
        "coverage_evaluator": {},
        "relevance_evaluator": {}
    }
    action_agent = ToTAgent(model="gpt-4o-2024-11-20", state_evaluator_configs=state_evaluator_configs, process_evaluator_configs=process_evaluator_configs)
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
        trajectory, summary, action_history, _ = await action_agent(query="The arena where the Lewiston Maineiacs played their home games can seat how many people?", fact=None)
        print(trajectory)
        print(summary)
        print(trajectory)
        response_agent = ResponseAgent(model="gpt-4o-2024-11-20")
        response = response_agent.response(query="The arena where the Lewiston Maineiacs played their home games can seat how many people?", messages=trajectory)
        print(response)
    finally:
        await action_agent.cleanup()
        await mcp_client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
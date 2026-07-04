import re
import json
from typing import List, Tuple, Any, Union
from urllib.parse import urlparse

from deep_search.utils.api import call_openai
from deep_search.utils.inspectors import TextInspectorTool, visualizer

class BaseStep:

    def __init__(self, raw_memory: str, role: str="assistant", model: str="gpt-4o-2024-11-20"):
        self.raw_memory = raw_memory
        self.role = role
        self.model = model

    def format_str(self) -> str:
        return self.raw_memory

    def format_prompt(self) -> str:
        return self.raw_memory

    def format_message(self, model: str) -> str:
        if "qwq" in model.lower() or "qwen" in model.lower() or "qwen3" in model.lower():
            return {
                "role": self.role,
                "content": self.format_prompt()
            }
        else:
            return {
                "role": self.role,
                "content": [{"type": "text", "text": self.format_prompt()}]
            }
    
    def __str__(self) -> str:
        return self.format_str()

class SystemStep(BaseStep):

    def __init__(self, system_prompt: str):
        super().__init__(system_prompt, role="system")
        self.system_prompt = system_prompt

class UserStep(BaseStep):

    def __init__(self, query: str, attachments: str=None, model: str="gpt-4o-2024-11-20"):
        self.query = query
        self.attachments = attachments
        self.model = model
        self.visualizer = visualizer
        self.text_inspector = TextInspectorTool(model=self.model, text_limit=1000)
        raw_str = self.format_prompt()
        super().__init__(raw_str, role="user")
        print("""-----------------UserStep----------------""", raw_str, sep="\n\n")

    def format_prompt(self) -> str:
        user_query = f"""Task: {self.query}"""
        if self.attachments:
            user_query += "\n\nTo solve the task above, you will have to use these attached files:\n"
            user_query += self.attachments
        return user_query
    
    def format_message(self, model: str) -> str:
        if "qwq" in model.lower() or "qwen" in model.lower() or "qwen3" in model.lower():
            return {
                "role": self.role,
                "content": self.format_prompt()
            }
        else:
            return {
                "role": self.role,
                "content": [{"type": "text", "text": self.format_prompt()}]
            }

    def _get_file_description(self, file_path: str) -> str:
        text_inspector = TextInspectorTool(model=self.model, text_limit=1000)
        return text_inspector(file_path)


class NoticeStep(BaseStep):

    def __init__(self, notice: str=None):
        if notice is None:
            notice = "好的，请生成一个sh脚本或生成一段python代码来完成你的动作。"
        self.notice = notice
        super().__init__(notice, role="user")
        print("""-----------------NoticeStep----------------""", self.format_prompt(), sep="\n\n")

class MissingInfoNoticeStep(BaseStep):

    def __init__(self, eval_results: List[dict]):
        self.eval_results = eval_results
        raw_str = self.format_prompt()
        super().__init__(raw_str, role="assistant")
        print("""-----------------MissingInfoNoticeStep----------------""", raw_str, sep="\n\n")

    def format_prompt(self) -> str:
        eval_results_str = ""
        count = 0
        for i, result in enumerate(self.eval_results):
            # eval_results_str += "\n".join([f"{i+1}. {aspect}" for i, aspect in enumerate(result["missing_aspects"], start=count)]) + "\n"
            # count += len(result["missing_aspects"])
            if result["is_satisfied"] != "yes":
                missing_aspects = "\n".join([f"\t{i+1}. {aspect}" for i, aspect in enumerate(result["missing_aspects"])])
                eval_results_str += f"""{count+1}. {result['reasoning']}需要继续考虑检索以下信息：
<missing_aspects>
{missing_aspects}
</missing_aspects>
"""
                count += 1

        return f"""为确保回答更完整，并且更符合用户需求，请考虑根据以下提示继续收集信息，这是必须的要求，请必须满足：
<hint>
{eval_results_str}
</hint>
用户并不期望你能看到这些提示，所以请不要在后续的生成中暴露提示的存在。
"""

class ObservationStep(BaseStep):

    def __init__(self, tool_name: str, observation: str, model: str="gpt-4o-2024-11-20"):
        super().__init__(observation, role="user")
        self.tool_name = tool_name
        self.observation = observation
        self.model = model
        print("""-----------------ObservationStep----------------""", self.format_prompt(), sep="\n\n")
        
    def format_message(self, model: str) -> str:
        # if "qwq" in model.lower() or "qwen" in model.lower():
        #     return [{"role": "tool", "name": self.tool_name, "content": self.observation}]
        # else:
        return [{"role": "user", "content": self.observation}]

class FactStep(BaseStep):

    def __init__(self, facts: str):
        super().__init__(facts, role="assistant")
        self.facts = facts
        print("""-----------------FactStep----------------""", self.format_prompt(), sep="\n\n")

class KeyPointsStep(BaseStep):

    def __init__(self, key_points: List[str]):
        self.key_points = key_points
        raw_str = self.format_prompt()
        super().__init__(raw_str, role="assistant")
        print("""-----------------KeyPointsStep----------------""", raw_str, sep="\n\n")

    def format_prompt(self) -> str:
        key_points_str = "\n".join([f"{i+1}. {point}" for i, point in enumerate(self.key_points)])
        return f"""For the user's question, I will consider collecting information based on the following key points, ensuring the content is complete, accurate, and meets the user's needs:
{key_points_str}
"""
    
    def get_key_points(self) -> List[str]:
        return self.key_points

class PlanStep(BaseStep):

    def __init__(self, query: str, fact: str, plan_json: dict):
        self.query = query
        self.fact = fact
        self.outlines = plan_json["outlines"]
        self.thought = plan_json["aspect"]
        raw_memory = self.build_prompt()
        super().__init__(raw_memory)
        print("""-----------------PlanStep----------------""", self.format_prompt(), sep="\n\n")

    def build_prompt(self) -> str:
        steps = "\n".join([f"{i+1}. {outline}" for i, outline in enumerate(self.outlines)])
        return f"""1. Plan Thought Process
Based on the user's question and fact analysis, the plan thought process is as follows:
{self.thought}

2. Plan Steps
Based on the plan thought process, the plan steps are as follows:
{steps}
"""

    def format_prompt(self) -> str:
        return self.raw_memory

class ActionStep(BaseStep):

    def __init__(self, thinking: str, tool_calls: list[dict], model: str):
        super().__init__(thinking)
        self.thinking = thinking
        self.tool_calls = tool_calls
        self.model = model
        print("""-----------------ActionStep----------------""", self.format_prompt(), sep="\n\n")

    def format_message(self, model: str) -> str:
        if "qwq" in model.lower() or "qwen" in model.lower() or "qwen3" in model.lower():
            return [{"role": "assistant", "content": self.thinking, "tool_calls": [{"type": "function", "function":{'name': tool_call["function"]["name"], 'arguments': tool_call["function"]["arguments"]}} for tool_call in self.tool_calls]}]
        else:
            tool_calls = "\n".join([f"Tool: {tool_call['function']['name']}\n Parameters: {tool_call['function']['arguments']}" for tool_call in self.tool_calls])
            return [{"role": "assistant", "content": self.thinking + "\n\n" + tool_calls}]

    def get_thinking(self) -> str:
        return self.thinking

    def get_tool_call(self) -> list[dict]:    
        return [{'name': tool_call["function"]["name"], 'arguments': tool_call["function"]["arguments"]} for tool_call in self.tool_calls]
    

class SearchStep(BaseStep):

    def __init__(self, search_results: list[dict]):
        self.search_results = search_results
        self.index_map = {result['url']: i+1 for i, result in enumerate(search_results)}
        raw_str = self.format_prompt()
        super().__init__(raw_str, role="user")
        print("""-----------------SearchStep----------------""", raw_str, sep="\n\n")

    def get_url_info(self) -> str:
        url_info = ""
        for i, result in enumerate(self.search_results, start=1):
            url_info += f"""
{i}. {result['name']}
    
    Host: {result['host_name']}

    URL: [{result['url']}]({result['url']})

    Snippet: {result['snippet']}

---
"""
        return url_info

    def format_prompt(self) -> str:
        prompt = f"""Below are examples of primary search results. If the original user query requires accessing a URL (e.g., to access the URL in a multi-hop manner and obtain images, etc.), please directly access the relevant URL instead of using unrelated URLs or images:

Search Results:
{self.get_url_info()}

Please note that web summaries are usually brief and not very specific.

- For specific information, you must use the ask_question_about_documents tool to query the URL or document to find the information you need.

- If you are looking for specific values, make sure to continue searching for reliable sources and accurate values, rather than using web summaries that may contain approximate values (e.g., the amount in the summary may be too inaccurate for the question)."""

        return prompt

    def update_idxs(self, history_idxs: dict) -> dict:
        history_idxs_len = len(history_idxs)
        reset_idxs = {}
        unique_urls = []
        for search_result in self.search_results:
            if search_result['url'] in history_idxs:
                reset_idxs[search_result['url']] = history_idxs[search_result['url']]
            else:
                unique_urls.append(search_result['url'])
                reset_idxs[search_result['url']] = history_idxs_len + len(unique_urls)
        return reset_idxs

class VisitStep(BaseStep):

    def __init__(self, url: str=None, file: str=None, file_type: str=None, title: str=None, content: str=None, max_length: int=2000, error_msg: str=None):
        self.url = url
        self.title = title
        self.file = file
        self.file_type = file_type
        self.content = content
        self.max_length = max_length
        self.error_msg = error_msg
        self.url_idx = None
        raw_str = self.format_prompt()
        super().__init__(raw_str, role="user")
        print("""-----------------VisitStep----------------""", raw_str, sep="\n\n")

    def format_prompt(self) -> str:
        if self.url:
            target_prompt = f"website {self.url}"
            if self.url_idx is not None:
                target_prompt += f" (number: {self.url_idx})"
        elif self.file:
            target_prompt = f"file {self.file}"
        
        if self.error_msg:
            return f"""Failed to visit {target_prompt}, the error message is:
            {self.error_msg}
            """
        if len(self.content) > self.max_length:
            # content_summarize = self.summarize(self.content)
            return f"""Visited {target_prompt}
The first {self.max_length} characters of {target_prompt} are:
{self.content[:self.max_length]}

The remaining {len(self.content) - self.max_length} characters are not displayed.

The undisplayed content of {target_prompt} contains the following information.

If you want to get specific information from this webpage, please use the ask_question tool to continue querying for specific details.
"""
        else:
            return f"""Visited {target_prompt}
The full content of {target_prompt} is:
{self.content}
            """
    
    def summarize(self, content: str) -> str:
        system_prompt = """You are a professional information summarization expert, skilled at accurately extracting paragraph structures from large amounts of text and comprehensively summarizing the core content of each paragraph. Your task is to ensure the completeness of the information, not omitting any key details, while making the summary clear, well-structured, and faithful to the original information."""
        user_prompt = f"""{content}"""
        response = call_openai('gpt-4o-mini-2024-07-18', [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        return response['content']

    def set_idx(self, url_map: dict):
        self.url_idx = url_map[self.url]

class QuesDocStep(BaseStep):

    def __init__(self, answer: str, tag: str = None, post_prompt: str = None, question: str = None, target_file: str = None, target_url: str = None):
        super().__init__(answer, role="user")
        self.tag = tag
        self.post_prompt = post_prompt
        self.target_file = target_file
        self.target_url = target_url
        self.question = question
        self.url_map = None
        print("""-----------------QuesDocStep----------------""", self.format_prompt(), sep="\n\n")

    def format_prompt(self) -> str:
        target_description = ""
        if self.target_file:
            target_description += "、".join([f"文件 {target_file} " for target_file in self.target_file])
            if self.target_url:
                target_description += "和 "
        if self.target_url:
            target_description += "、".join([f"网页 {target_url} " for target_url in self.target_url])

        prompt = f"""Question:
{self.question}

Based on the information contained in {target_description}, the following answer was obtained:
"""
        # prompt += f'ENDOFTURN\n'
        # if self.tag:
        #     prompt += f'<{self.tag}>\n'
        prompt += self.raw_memory + '\n'
        # if self.tag:
        #     prompt += f'\n</{self.tag}>\n'
        # prompt += f'\nENDOFTURN\n'
        if self.url_map:
            prompt += f"""
The following is the index of the webpage:
{self.url_map}
"""
        if self.post_prompt:
            prompt += self.post_prompt + '\n'
        return prompt

    def set_idx(self, index_map: dict):
        url_map_str = ""
        for url in self.target_url:
            url_map_str += f"{url} : {index_map[url]}\n"
        self.url_map = url_map_str

class CodeStep(BaseStep):

    def __init__(self, code: str, res_type: str, text: str, image: str, error_msg: str):
        self.code = code
        self.res_type = res_type
        self.text = text
        self.image = image
        self.error_msg = error_msg
        raw_str = self.format_prompt()
        super().__init__(raw_str, role="user")
        print("""-----------------CodeStep----------------""", raw_str, sep="\n\n")
    
    def format_prompt(self) -> str:
        prompt = f"""The following code was executed:
        ## code
        {self.code}
        """

        if self.error_msg:
            prompt += f"""The code execution failed, and the error message is:
            {self.error_msg}
            """
        elif self.res_type == "text":
            prompt += f"""The code execution was successful, and the result is:
            {self.text}
            """
        elif self.res_type == "image":
            prompt += f"""The code execution was successful, and the result is an image, which is stored in {self.image}
            """

        return prompt

class ToolStep(BaseStep):

    def __init__(self, tool_name: str, tool_arguments: Union[str, dict]):
        super().__init__(f"Used tool: {tool_name}\n", role="user")
        self.tool_name = tool_name
        self.tool_arguments = json.loads(tool_arguments) if isinstance(tool_arguments, str) else tool_arguments
        print("""-----------------ToolStep----------------""", self.format_prompt(), sep="\n\n")

    def format_prompt(self) -> str:
        return f"""Used tool: {self.tool_name} 
Parameters: {self.tool_arguments}"""

    def get_name(self) -> str:
        return self.tool_name

    def get_params(self) -> dict:
        return self.tool_arguments

class TaskStep(BaseStep):

    def __init__(self, task: str, summary: str, trajectory: str, search_steps: List[str]):
        super().__init__(task, role="user")
        self.task = task
        self.summary = summary
        self.trajectory = trajectory
        self.search_steps = search_steps
        print("""-----------------TaskStep----------------""", self.format_prompt(), sep="\n\n")

    def format_prompt(self) -> str:
        search_list = "\n".join(self.search_steps)
        return f"""{self.task}

# Subtask execution record
{self.summary}"""
        
class StepManager:

    def __init__(self, model: str, agent_work_dir: str):
        self.step_list: List[BaseStep] = []
        self.model = model
        self.agent_work_dir = agent_work_dir
        self.history_idxs = {}
        self.archive = None
        self.key_points = None

    def set_key_points(self, key_points: KeyPointsStep):
        self.key_points = key_points

    def is_info_step(self, step: BaseStep) -> bool:
        return isinstance(step, (VisitStep, QuesDocStep, CodeStep, ObservationStep, TaskStep))

    def format_search_info(self) -> str:
        try:
            return "\n".join([step.format_prompt() for step in self.step_list if self.is_info_step(step)])
        except Exception as e:
            import pdb; pdb.set_trace()

    def add_step(self, step: BaseStep):
        if isinstance(step, SearchStep):
            reset_idxs = step.update_idxs(self.history_idxs) 
            self.history_idxs.update(reset_idxs)
        elif isinstance(step, (QuesDocStep, VisitStep)):
            step.set_idx(self.history_idxs)
        self.step_list.append(step)
        # if self.is_info_step(step):
        #     self.archive = self._update_archive(step)

    def add_step_list(self, step_list: List[BaseStep]):
        for step in step_list:
            self.add_step(step)

    def _update_archive(self, step: BaseStep):
        if self.archive is None:
            return self._organize_search_info()
        else:
            return self._supplement_search_info(step)

    def _organize_search_info(self):
        # TODO: 需要更好的组织搜索信息
        system_prompt = """You are a professional information organization expert, skilled at decomposing and integrating massive information. Please extract the core content based on the following search results and key points, and supplement the key content and the search result number for each key point.
Specific requirements:
Ensure the completeness and accuracy of information: Ensure that the extracted content is faithful to the search results, avoid subjective speculation or information loss.
Organize information in a structured manner according to the key points: Each key point should have clear core content and be expanded based on the search results.
Reference search result number: Clearly mark the reference search result number for each补充内容,以便追溯来源。
Remove redundant information: Avoid duplicate or irrelevant content, ensure information is concise and efficient.
Maintain logical clarity: If multiple search results provide different perspectives or conflicting information, summarize and note the differences.
Use professional and objective expressions: Ensure language is precise, objective, and does not introduce subjective evaluations or unverified information."""
        user_prompt = f"""
Search results:
{self.format_search_info()}

Key points information:
{self.key_points.format_str()}
"""
        response = call_openai(self.model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        return response['content']

    def _supplement_search_info(self, step: BaseStep):
        system_prompt = """You are a professional information organization expert, skilled at decomposing and integrating massive information. Please extract the core content of the search results based on the following search results and key points information, and supplement the appropriate content for the key points and the reference search result number.

Your task:
Supplement section content: Improve existing key points based on search results and add corresponding search result numbers.
Expand key points: Identify potential information gaps, set reasonable key points, and attach corresponding search result numbers.
Information accurate matching: Ensure that each supplement or new content is based on authoritative information and correctly references the search result numbers.
Please follow these requirements to expand and optimize the key points.
        """
        user_prompt = f"""
Key points information:
{self.archive}

Supplement section content:
{step.format_str()}"""
        response = call_openai(self.model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        return response['content']

    def build_action_history(self) -> str:
        # TODO: 需要更好的组织action history
        action_history = []
        for step in self.step_list:
            message = step.format_message(self.model)
            if isinstance(message, list):
                action_history.extend(message)
            else:
                action_history.append(message)
        return action_history
    
    def get_search_list(self) -> List[str]:
        return [step.format_prompt() for step in self.step_list if (isinstance(step, ObservationStep) and step.tool_name == "search")]

    def clear(self):
        self.step_list = []
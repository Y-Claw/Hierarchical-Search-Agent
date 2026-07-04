import re
from typing import List, Optional, Dict

from deep_search.utils.memory import PlanStep, FactStep, KeyPointsStep
from deep_search.utils.api import call_openai, gpt_json_load
from deep_search.utils.retry import retry

class PlanAgent:

    def __init__(self, model, agent_work_dir):
        self.model = model
        self.agent_work_dir = agent_work_dir
        self.meta_query_elements = []
        self.fact_history = []
        self.planning_history = [] # a list of meta query elements
        
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

    def build_planning_message(self, task, facts, history, attachments_description):
        if attachments_description:
            attachments_description = f"""
    The following is the attachment description:
```
{attachments_description}
```
"""
        else:
            attachments_description = """"""
        if history:
            pass
        else:
            return [{"role": "user",
                    "content": f"""You are a world-class planning expert capable of solving any task.

Now, for the given task, consider the above input and the list of facts to create a step-by-step high-level plan.
This plan should be based on the available tools and include various subtasks. If these tasks are executed correctly, the correct answer will be obtained.
Do not skip steps or add any unnecessary steps. Just write out the high-level plan without detailing specific tool calls.
Do not add any details from your own knowledge.

Task:
```
{task}
```

Known facts:
```
{facts}
```
{attachments_description}

You can use the following tools:
1. `search`：Execute Bing search and get results
2. `visit_urls`：View the text content of the first n characters of the provided URL
3. `ask_urls`：Ask a question to a specific URL to get relevant information (only the text content of the current URL, unable to page or jump)
4. `all_information_sufficient`：Collect information complete, exit current task

Output content:
1. thought: Carefully review the task requirements, think about how to solve this problem, your thinking process
2. plan: How to plan the solution steps (you can use the search tool), output the detailed steps of each step
""" + """
Please output your thought and solution plan in JSON format
- The output result only contains `thought` and `plan`
- `plan` is a dict, the key starts from `step1` and increases incrementally, representing step 1, step 2..., and the value of each part is in markdown format
- Example: {\"thought\": \"**\", \"plan\": {\"step1\": "***", \"step2\": "***", ...}} """}]

    def build_fact_extraction_message(self, task, history, attachments_description):
        if attachments_description:
            attachments_description = f"""
    The following is the attachment description:
    ```
    {attachments_description}
    ```
"""
        else:
            attachments_description = """"""
        if self.fact_history:
            pass
        else:
            return [{"role": "user", "content": f"""I will give you a task.

Now, you need to establish a comprehensive preparatory investigation to understand the facts we have already learned and the facts we still need to learn.
To do this, you need to read the task and determine the content that must be discovered to successfully complete the task.
Do not make any assumptions. For each item, please provide detailed reasoning. The following is the structure of the investigation:

### 1. Facts given in the task
List the specific facts that may be helpful in the task (there may be nothing here).

### 2. Facts to be found
List any facts we may need to find.
Also list where these facts can be found, such as websites, files, etc. - maybe the task contains some sources you should repeat here.

### 3. Facts to be derived
List any content we want to derive from the above content through logical reasoning, such as calculation or simulation.

Remember, "facts" usually refer to specific names, dates, numbers, etc. Your response should use the following titles:
### 1. Facts given in the task
### 2. Facts to be found
### 3. Facts to be derived
Do not add any other content.

Here is the task:
{task}
"""}]

    def _fact_extraction(self, task, history=None, attachments_description=None):
        messages = self.build_fact_extraction_message(task, history, attachments_description)
        response = call_openai(self.model, messages)
        # Update token usage
        self._update_token_usage(response.get("usage", {}))
        content = response["content"]
        self.fact_history.append(content)
        return content

    def fact_extraction(self, task, history=None, attachments_description=None):
        facts = self._fact_extraction(task, history, attachments_description)
        return FactStep(facts)

    def plan(self, task, history=None, attachments_description=None):
        facts = self._fact_extraction(task, history, attachments_description)
        messages = self.build_planning_message(task, facts, history, attachments_description)
        response = call_openai(self.model, messages)
        # Update token usage
        self._update_token_usage(response.get("usage", {}))
        content = response["content"]
        plan_json = self.parse_response(content)
        return [FactStep(facts), PlanStep(task, facts, plan_json)]

    def parse_response(self, response):
        json_res = gpt_json_load(response)
        rebuild_json = {"aspect": json_res["thought"], "outlines": [""] * len(json_res["plan"])}
        assert "thought" in json_res and "plan" in json_res and isinstance(json_res["plan"], dict)
        for k, v in json_res["plan"].items():
            pattern = r"step(\d+)"
            match = re.match(pattern, k)
            if match:
                index = int(match.group(1)) - 1  # 获取括号中的数字
                rebuild_json["outlines"][index] = v
        for o in rebuild_json["outlines"]:
            assert o
        return rebuild_json

class KeyPointsExtractor(object):
    def __init__(self, model: Optional[str] = None, time: Optional[str] = None) -> None:
        self.model = model
        self.time = time
        
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

    def system_prompt(self) -> str:
        system_prompt = """You are a professional information analysis assistant,擅长从用户查询中提取关键要点并以结构化方式呈现。你的任务是分析用户的查询，识别并提取与查询相关的关键要点，然后以JSON格式返回这些要点。

## Output Requirements
1. You must return all key points in JSON format
2. The JSON structure must contain the following fields:
   - "query": The original user query
   - "key_points": An array of key points, each point is an object containing:
     - "point": A concise description of the key point
     - "explanation": A detailed explanation of the point (optional)
     - "importance": A score of importance (1-5, 5 being the most important)
   - "summary": A concise summary of all key points

## Analysis Guidelines
- Identify the core concepts and keywords in the query
- Determine the main purpose and potential sub-problems of the query
- Consider different angles and dimensions of relevant points
- Sort key points by importance
- Ensure that points are logically coherent and mutually complementary
- Avoid repeating or overly broad points

## Example Output
User query: "The impact of AI on the labor market"

```json
{
  "query": "The impact of AI on the labor market",
  "key_points": [
    {
      "point": "Automation leads to the reduction of specific job positions",
      "explanation": "AI and automation technologies may replace repetitive, predictable jobs, such as certain positions in manufacturing, data processing, and customer service.",
      "importance": 5
    },
    {
      "point": "Creating new job opportunities",
      "explanation": "AI development has created new jobs, such as AI trainers, AI ethics experts, and data scientists.",
      "importance": 4
    },
    {
      "point": "Skill demand changes",
      "explanation": "The demand for technical skills, creative thinking, and emotional intelligence in the labor market increases.",
      "importance": 4
    },
    {
      "point": "Work性质变革",
      "explanation": "Many existing jobs will be redefined, and humans will focus on tasks that AI cannot easily complete.",
      "importance": 3
    },
    {
      "point": "Labor market polarization",
      "explanation": "It may exacerbate the income gap between high-skilled and low-skilled workers.",
      "importance": 3
    }
  ],
  "summary": "The impact of AI on the labor market is multi-faceted, including the reduction of certain job positions, the emergence of new jobs, changes in skill demand, changes in work性质, and possibly labor market polarization. Adaptation to these changes requires educational system reform and lifelong learning."
}
```

Remember, your response must always be in the above JSON format, do not add any other text, explanation, or markers. Return the correctly formatted JSON object directly.
"""
        return self.build_message("system", system_prompt)
    
    def user_prompt(self, prompt: str) -> str:
        user_prompt = f"""Please analyze the following user query and extract the key points in JSON format:
{prompt}
"""
        return self.build_message("user", user_prompt)
    
    def build_message(self, role:str, content:str) -> List[Dict[str, str]]:
        messages = [
            {"role": role, "content": content}
        ]
        return messages
    
    @retry(3, 1)
    def gpt_response(self, messages:List[Dict]) -> str:
        # think, response = call_ds_r1(messages)
        response = call_openai(self.model, messages)
        # Update token usage
        self._update_token_usage(response.get("usage", {}))
        think = response["content"]
        content = response["content"]
        response, summary = self.parse_gpt_response(content)
        return response, summary

    def parse_gpt_response(self, json_str:str) -> Dict:
        try: 
            response = gpt_json_load(json_str)
            assert "key_points" in response and "summary" in response
            summary = response["summary"]
            response = response["key_points"]
            for _ in response:
                assert "point" in _ and "explanation" in _ and "importance" in _
            return response, summary
            
        except Exception as e:
            raise ValueError(f"EvalPrompt parse_response{json_str} Error: {e}")

    def __call__(self, prompt: str) -> List[str]:
        messages = self.system_prompt() + self.user_prompt(prompt)
        response, summary = self.gpt_response(messages)
        print(response)
        keypoints = []
        for _ in response:
            keypoints.append(f"{_['point']}: {_['explanation']}")
        return KeyPointsStep(keypoints), summary


if __name__ == "__main__":
    prompt = "全球已知濒危动物的完整名单"
    model = "public-deepseek-r1"
    key_points_extractor = KeyPointsExtractor(model)
    output = key_points_extractor(prompt)
    print(output)


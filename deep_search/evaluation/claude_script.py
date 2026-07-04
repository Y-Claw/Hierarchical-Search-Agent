import json
from typing import List, Optional, Dict
import boto3
from botocore.config import Config
import functools
import asyncio
import os
import time
from copy import deepcopy
from tqdm import tqdm

def retry(max_retries, time_sleep):
    def decorator_retry(func):
        @functools.wraps(func)
        async def async_wrapper_retry(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_retries:
                        import traceback
                        traceback.print_exc()
                        print(f"All {max_retries} attempts failed: {e}")
                        raise ValueError(f"All {max_retries} attempts failed")
                    await asyncio.sleep(time_sleep)  # 使用异步睡眠

        @functools.wraps(func)
        def sync_wrapper_retry(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_retries:
                        import traceback
                        traceback.print_exc()
                        print(f"All {max_retries} attempts failed: {e}")
                        raise ValueError(f"All {max_retries} attempts failed")
                    time.sleep(time_sleep)

        # 根据函数是否是协程函数返回对应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper_retry
        return sync_wrapper_retry

    return decorator_retry

THINK_TOOL = {
            "name": "think",
            "description": "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                "thought": {
                    "type": "string",
                    "description": "A thought to think about."
                }
            },
            "required": ["thought"]
        }
    }

@retry(max_retries=3, time_sleep=1)
def call_aws_claude(messages: List[Dict[str, str]], system: Optional[str] = None, tools: Optional[List[Dict[str, str]]] = None, think_tool=False):
    kwargs = {}
    kwargs['anthropic_version'] = "bedrock-2023-05-31"
    kwargs['max_tokens'] = 128000
    kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": 4000
        }

    if system:
        kwargs['system'] = system

    if tools:
        kwargs['tools'] = deepcopy(tools)
        if think_tool:
            kwargs['tools'].append(THINK_TOOL)

    kwargs['messages'] = messages

    body = json.dumps(kwargs)
    proxy = os.getenv("BEDROCK_PROXY")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    config = Config(
        proxies=proxies,
        retries={
            'max_attempts': 5,
            'mode': 'adaptive'
        }
    )
    bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                config=config
            )
    response = bedrock_client.invoke_model(
        body=body, 
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    )
    response_body = json.loads(response.get("body").read())

    return response_body


def multi_step_call(query: str, system: Optional[str] = None):
    # initial env and tools
    from search_toolkit.environments.browser_environment import BrowserEnvironment
    proxy = os.getenv("BROWSER_PROXY")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    toolkit = BrowserEnvironment(proxies=proxies)
    functions = toolkit.get_tools()
    claude_functions = [
        {
            "name": function["name"],
            "description": function["description"],
            "input_schema": function["parameters"]
        }
        for function in functions
    ]

    # initial messages and setting
    messages = [
        {"role": "user", "content": query}
    ]
    think_tool = True
    max_steps = 30
    step = 0

    while step < max_steps:
        step += 1
        # call claude
        response = call_aws_claude(messages, system, claude_functions, think_tool)
        # three type in content: thinking, tool_use, text
        messages.append({"role": "assistant", "content": response["content"]})
        print(response["content"], end="\n=======\n")

        # call tool
        tool_call_result = []
        for part in response["content"]:
            if part["type"] == "tool_use":
                tool_name = part["name"]
                tool_input = part["input"]
                tool_id = part["id"]
                if tool_name != "think":
                    tool_result = toolkit.call_tool(tool_name, tool_input)
                    tool_call_result.append({"type": "tool_result", "tool_use_id": tool_id, "content": tool_result})
                else:
                    tool_call_result.append({"type": "tool_result", "tool_use_id": tool_id})

        if tool_call_result:
            messages.append({"role": "user", "content": tool_call_result})
        else:
            break
    return messages

if __name__ == "__main__":
    os.environ.setdefault("JINA_TRANSFER_URL", "http://localhost:8000")
    os.environ["TOOL_TIMEOUT"] = "60"
    os.environ["DOWNLOAD_DIR"] = "./downloads"

    max_cnt = 100
    cnt = 0
    path = "/cloud/ai_search/raw_data/ObjectiveQA/browsecomp.jsonl"
    with open(path, "r") as f:
        with open("claude_log.jsonl", "w") as f_out:
            for line in tqdm(f, total=max_cnt):
                data = json.loads(line)
                query = data["question"]
                answer = data["answer"]
                claude_log = multi_step_call(query)
                data["claude_log"] = claude_log
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

                if cnt >= max_cnt:
                    break
                cnt += 1

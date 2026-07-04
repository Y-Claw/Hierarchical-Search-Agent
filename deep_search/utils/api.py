from enum import Enum
from dotenv import load_dotenv
import json
import requests
import concurrent.futures
from transformers import AutoTokenizer
import re

from openai import OpenAI
import httpx
import os
from deep_search.utils.retry import retry
from deep_search.utils.config import QWQ_CKPT_PATH, QWEN_CKPT_PATH

load_dotenv()

def allow_system(model):
    return model not in ["o1-mini-2024-09-12", "o1-preview"]

def allow_max_tokens(model):
    return model not in ["o1-mini-2024-09-12", "o1-preview"]

fixed_temperature = {
    "o1-mini-2024-09-12": 1,
    "o1-preview": 1,
    "gpt-5-mini-2025-08-07": 1,
    "gpt-5-nano-2025-08-07": 1,
}

def gpt_json_load(json_str):
    json_str = json_str.replace("```json", "").replace("```", "").replace("\n", "").replace("None", "null").strip()
    try:
        json_data = json.loads(json_str, strict=False)
    except Exception as e:
        first_bracket_idx = json_str.index("{")
        last_bracket_idx = json_str.rindex("}") + 1
        json_str = json_str[first_bracket_idx:last_bracket_idx]
        json_data = json.loads(json_str, strict=False)
    if isinstance(json_data, str):
        return json.loads(json_data)
    return json_data

def prepare_completion_params(model, messages, temperature, response_format, max_tokens):
    if not allow_system(model):
        for m in messages:
            if m["role"] == "system":
                m["role"] = "user"
    if not allow_max_tokens(model):
        max_tokens = None
    if model in fixed_temperature:
        temperature = fixed_temperature[model]
    return messages, temperature, response_format, max_tokens

OPENAI_MODELS = ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "gpt-4o-2024-02-15", "o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07", "openrouter:gpt-4o-mini-2024-07-18"]
CLAUDE_MODELS = ["claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219-thinking"]
QWEN_MODELS = ["Qwen2.5-32B-Instruct-128k", "Qwen2.5-14B-Instruct-128k", "Qwen2.5-14B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct"]
DS_R1_MODELS = ["public-deepseek-r1"]

QWEN_ENDPOINTS = {
    model: endpoint for model, endpoint in {
        "Qwen3-32B": os.getenv("QWEN3_32B_API_BASE"),
        "Qwen2.5-32B-Instruct": os.getenv("QWEN_32B_API_BASE"),
        "Qwen2.5-14B-Instruct": os.getenv("QWEN_14B_API_BASE"),
        "Qwen2.5-14B-Instruct-128k": os.getenv("QWEN_14B_128K_API_BASE"),
        "Qwen2.5-7B-Instruct": os.getenv("QWEN_7B_API_BASE"),
    }.items() if endpoint
}

@retry(3, 1)
def call_openai(model, messages, temperature=0.7, max_tokens=8192, response_format=None, **kwargs):
    if model in OPENAI_MODELS or model in CLAUDE_MODELS:
        return openai_model(model, messages, temperature, max_tokens, response_format, **kwargs)
    elif model in DS_R1_MODELS:
        return ds_r1(messages, temperature, show_stream=True)
    elif model in QWEN_MODELS:
        return qwen_model(model, messages, temperature, max_tokens, response_format, **kwargs)
    else:
        return call_deploy(model, messages, temperature, max_tokens, response_format, **kwargs)
    

def openai_model(model, messages, temperature=0.7, max_tokens=None, response_format=None, **kwargs):
    proxy = kwargs.pop("proxy", None)
    proxy = proxy or os.getenv("OPENAI_PROXY")
    client = OpenAI(
            timeout=120,
            max_retries=3,
            base_url=os.getenv("ONE_API_BASE", "https://api.openai.com/v1"),
            api_key=os.getenv("ONE_API_KEY"),
            http_client=httpx.Client(
                proxy=proxy,
            )
        )
    messages, temperature, response_format, max_tokens = prepare_completion_params(model, messages, temperature, response_format, max_tokens)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        # max_tokens=max_tokens,
        **kwargs
    )
    tool_calls = []
    if response.choices[0].message.tool_calls:
        for item in response.choices[0].message.tool_calls:
            item_dict = item.to_dict()
            item_dict["function"]["arguments"] = json.loads(item_dict["function"]["arguments"])
            tool_calls.append(item_dict)
    
    # 添加 token 使用信息
    usage_info = {}
    if hasattr(response, 'usage') and response.usage:
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    
    return {
        "think": "", 
        "content": response.choices[0].message.content, 
        "tool_calls": tool_calls, 
        "raw_message": response.choices[0].message.to_dict(),
        "usage": usage_info
    }

def ds_r1(messages, temperature=0.7, show_stream=False):
    key = os.getenv("MODELNEST_API_KEY")
    url = os.getenv("MODELNEST_API_URL")
    client = OpenAI(
        timeout=120,
        max_retries=3,
        base_url=url,
        api_key=key,
    )
    stream = client.chat.completions.create(
        messages=messages,
        model="public-deepseek-r1",
        temperature=temperature,
        stream=True
    )
    content = ""
    for part in stream:
        part_str = part.choices[0].delta.content or ""
        content += part_str
        if show_stream:
            print(part_str, end="", flush=True)
    end_think_str = "</think>"
    end_think_idx = content.index(end_think_str) + len(end_think_str)
    return {
        "content": content[end_think_idx:].strip(), 
        "think": content[:end_think_idx].strip(), 
        "role": "assistant",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@retry(3, 1)
def call_deploy(model, messages, temperature=0.7, max_tokens=8192, response_format=None, **kwargs):
    base_url =  os.getenv("DEPLOY_API_BASE") 
    tool_choice = kwargs.pop("tool_choice", "auto")
    tokenizer = AutoTokenizer.from_pretrained(QWQ_CKPT_PATH)
    if "tools" in kwargs:
        text = tokenizer.apply_chat_template(messages, tools=kwargs["tools"], add_generation_prompt=True, tokenize=False)
    else:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    response = requests.post(f"{base_url}/completions", json={"model": model, "prompt": text, "temperature": temperature, "max_tokens": max_tokens, "response_format": response_format})
    if "<think>" not in response.json()["choices"][0]["text"]:
        response_json = "<think>\n" + response.json()["choices"][0]["text"]
    else:
        response_json = response.json()["choices"][0]["text"]
    response_messages = try_parse_tool_calls(response_json)
    response_messages["think"], response_messages["content"] = split_think_content(response_messages["content"])
    response_messages["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # while "tools" in kwargs and tool_choice == "required" and "tool_calls" not in response_messages:
    #     think = response_messages["think"].replace("</think>", "\n\n") + response_messages["content"] + "\n</think>\n\n<tool_call>\n"
    #     response_messages["content"] = think
    #     addition_messages = messages + [response_messages]
    #     text = tokenizer.apply_chat_template(addition_messages, tools=kwargs["tools"], tokenize=False, continue_final_message=True)
    #     addition_response = requests.post(f"{base_url}/completions", json={"model": model, "prompt": text, "temperature": temperature, "max_tokens": max_tokens, "response_format": response_format})
    #     addition_json = addition_response.json()
    #     if "</tool_call>" not in addition_json["choices"][0]["text"]:
    #         continue
    #     addition_json["choices"][0]["text"] = think + addition_json["choices"][0]["text"]
    #     response_messages = try_parse_tool_calls(addition_json["choices"][0]["text"])
    #     response_messages["think"], response_messages["content"] = split_think_content(response_messages["content"])
    return response_messages

@retry(3, 1)
def qwen_model(model, messages, temperature=0.7, max_tokens=None, response_format=None, **kwargs):
    if model in QWEN_ENDPOINTS:
        base_url = QWEN_ENDPOINTS[model]
    else:
        base_url =  os.getenv("QWEN_DEPLOY_API_BASE") 
    tool_choice = kwargs.pop("tool_choice", "auto")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_CKPT_PATH)
    if "tools" in kwargs:
        text = tokenizer.apply_chat_template(messages, tools=kwargs["tools"], tool_choice=kwargs.get("tool_choice", "auto"), add_generation_prompt=True, tokenize=False)
    else:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    response = requests.post(f"{base_url}/completions", json={"model": model, "prompt": text, "temperature": temperature, "max_tokens": max_tokens, "response_format": response_format})
    try:
        response_json = response.json()["choices"][0]["text"]
    except Exception as e:
        import pdb; pdb.set_trace()
    response_messages = try_parse_tool_calls_qwen(response_json, thinking=False)
    response_messages["think"] = ""
    response_messages["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return response_messages

def split_think_content(content: str):
    end_think_str = "</think>"
    # if end_think_str not in content:
    #     return "", content
    end_think_idx = content.index(end_think_str) + len(end_think_str)
    return content[:end_think_idx].strip(), content[end_think_idx:].strip()

def try_parse_tool_calls_qwen(content: str, thinking=True):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    if thinking:
        think, response = split_think_content(content)
    else:
        think, response = "", content
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+?)\n</tool_call>", response, re.DOTALL)):
        if i == 0:
            offset = len(think) + m.start()
        try:
            response_text = m.group(1)
            left_idx = response_text.index("{")
            right_idx = response_text.rindex("}") + 1
            func = json.loads(response_text[left_idx:right_idx])
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else: 
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content), "tool_calls": []}


def try_parse_tool_calls(content: str, thinking=True):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    if thinking:
        think, response = split_think_content(content)
    else:
        think, response = "", content
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+?)\n</tool_call>", response, re.DOTALL)):
        if i == 0:
            offset = len(think) + m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else: 
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content), "tool_calls": []}

def fetch_single_url_content(type, url):
    headers = {
        "Content-Type": "application/json"
    }
    try:
        data = {
            "url": url
        }
        response = requests.post("http://10.51.181.205:8000/fetch", json=data, headers=headers)
        return url, response.json().get(f"{type}_content", "")
    except Exception as e:
        return url, ""

def qinyan_fetch_content(url_list, type="pa"):
    assert type in ["pa", "md"]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(fetch_single_url_content, type, url): url for url in url_list}
        for future in concurrent.futures.as_completed(future_to_url):
            url, content = future.result()
            results[url] = content   
    return results

if __name__ == "__main__":
    # 测试不同模型的token使用信息返回
    print("=== 测试OpenAI模型 (真实token使用信息) ===")
    try:
        result = call_openai("gpt-4o-mini-2024-07-18", [{"role": "user", "content": "Hello, how are you?"}])
        print(f"Content: {result['content']}")
        print(f"Usage: {result['usage']}")
    except Exception as e:
        print(f"OpenAI模型测试失败: {e}")
    
    print("\n=== 测试Qwen模型 (0占位符) ===")
    try:
        result = call_openai("Qwen2.5-14B-Instruct-128k", [{"role": "user", "content": "Hello, how are you?"}])
        print(f"Content: {result['content']}")
        print(f"Usage: {result['usage']}")
    except Exception as e:
        print(f"Qwen模型测试失败: {e}")
    
    print("\n=== 测试DeepSeek R1模型 (0占位符) ===")
    try:
        result = call_openai("public-deepseek-r1", [{"role": "user", "content": "Hello, how are you?"}])
        print(f"Content: {result['content']}")
        print(f"Usage: {result['usage']}")
    except Exception as e:
        print(f"DeepSeek R1模型测试失败: {e}")
    
    print("\n=== 测试工具调用功能 ===")
    kwargs = {
        "tool_choice": "required",
        "tools": [
            {"type": "function", "function": {"name": "weather", "description": "查询指定地点的天气", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "查询地点"}}, "required": ["location"]}}}
        ]
    }
    try:
        result = call_openai("gpt-4o-mini-2024-07-18", [{"role": "user", "content": "帮我查询北京的天气"}], **kwargs)
        print(f"Content: {result['content']}")
        print(f"Tool calls: {result['tool_calls']}")
        print(f"Usage: {result['usage']}")
    except Exception as e:
        print(f"工具调用测试失败: {e}")

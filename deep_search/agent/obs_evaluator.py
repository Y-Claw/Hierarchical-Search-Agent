import json
from typing import List, Optional, Dict
from datetime import datetime

from deep_search.utils.api import call_openai, gpt_json_load
from deep_search.utils.retry import retry


class EvalObservation(object):
    def __init__(self, model: Optional[str] = None, time: Optional[str] = None) -> None:
        self.model = model
        self.time = time

    def system_prompt(self) -> str:
        system_prompt = """你是一个专业的信息评估助手，擅长分析搜索结果是否满足特定回答要点的需求。你的任务是判断给定的搜索信息(observation)是否满足用户查询(query)对应的特定回答要点，并提供详细的判断理由。

## 评估任务
1. 仔细分析用户的查询(query)和对应的回答要点
2. 详细阅读提供的搜索信息(observation)
3. 判断搜索信息是否包含足够内容来满足该回答要点
4. 给出详细的判断理由，包括支持或不支持的具体证据

## 输出要求
你必须以JSON格式返回评估结果，包含以下字段：
- "query": 用户的原始查询
- "key_point": 需要评估的回答要点
- "observation": 搜索到的信息摘要(限制在100字以内)
- "is_satisfied": "yes" 或者 "no"
- "confidence": 置信度评分(1-5，5为最高)
- "reasoning": 详细的判断理由，包括关键证据和逻辑分析
- "missing_aspects": 如果不满足，列出缺失的关键信息(数组格式)

## 评估标准
- **完全满足(true, 置信度4-5)**: 搜索信息直接回答了要点，提供了具体、准确、全面的相关信息
- **部分满足(true, 置信度2-3)**: 搜索信息包含部分相关内容，但不够全面或深入
- **不满足(false)**: 搜索信息与要点无关，或信息不足以支持要点

## 示例输出
```json
{
  "query": "人工智能对医疗行业的影响",
  "key_point": "AI在医学影像诊断中的应用",
  "observation": "研究表明，基于深度学习的AI系统在X光片和CT扫描分析中准确率达到95%，比普通放射科医生高出7%...",
  "is_satisfied": "yes",
  "confidence": 5,
  "reasoning": "搜索信息直接讨论了AI在医学影像(X光和CT)诊断中的应用，并提供了具体的准确率数据(95%)和与人类医生的比较(高出7%)，这些都是关于AI在医学影像诊断应用的具体证据。",
  "missing_aspects": []
}
```

```josn
{
  "query": "人工智能对医疗行业的影响",
  "key_point": "AI在医学影像诊断中的应用",
  "observation": "医疗行业正在经历数字化转型，各大医院投资升级IT系统，改善患者体验和医疗记录管理...",
  "is_satisfied": "no",
  "confidence": 1,
  "reasoning": "搜索信息只讨论了医疗行业的数字化转型和IT系统升级，没有提及AI技术在医学影像诊断中的任何应用或相关技术细节。",
  "missing_aspects": ["AI与医学影像的关联", "诊断准确率数据", "具体应用案例"]
}
```

请记住，你的回答必须始终采用上述JSON格式，不要添加任何其他文本、解释或标记。直接返回格式正确的JSON对象。
"""
        return self.build_message("system", system_prompt)
    
    def user_prompt(self, prompt: str, keypoint: str, observations: str) -> str:
        user_prompt = f"""### 用户查询（query）
{prompt}

### 回答要点（keypoint）
{keypoint}

### 搜索到的信息（observation）
{observations}
"""
        return self.build_message("user", user_prompt)
    
    def build_message(self, role:str, content:str) -> List[Dict[str, str]]:
        messages = [
            {"role": role, "content": content}
        ]
        return messages
    
    @retry(3, 1)
    def gpt_response(self, messages:List[Dict]) -> str:
        response = call_openai(self.model, messages)["content"]
        response = self.parse_gpt_response(response)
        return response

    def parse_gpt_response(self, json_str:str) -> Dict:
        try:
            response = gpt_json_load(json_str)
            assert "is_satisfied" in response and "confidence" in response
            assert isinstance(response["is_satisfied"], str) and isinstance(response["confidence"], int)
            return response
        except Exception as e:
            raise ValueError(f"EvalObs parse_response{json_str} Error: {e}")

    def __call__(self, prompt: str, keypoints: List[str], observations: List[str],) -> Dict:
        observations = "\n\n".join(observations)
        eval_results = []
        for keypoint in keypoints:
            messages = self.system_prompt() + self.user_prompt(prompt, keypoint, observations)
            response = self.gpt_response(messages)
            eval_results.append(response)
        is_enough = all(result.get("is_satisfied", "no") in ["yes", "Yes", "YES"] for result in eval_results) 
        return is_enough, eval_results

if __name__ == '__main__':
    prompt = "滑雪板推荐"
    model = "public-deepseek-r1"
    Obs_Eval = EvalObservation(model)
    # Prompt_Eval = EvalPrompt(model)
    output = Obs_Eval(prompt)
    print(output)

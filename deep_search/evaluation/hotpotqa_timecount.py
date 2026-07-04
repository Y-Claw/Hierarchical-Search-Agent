import re
import os
import json
import concurrent
import time
import random
import asyncio
from datetime import datetime
from tqdm import tqdm
import argparse
import pandas as pd
import yaml

from datasets import load_dataset
import openai

from deep_search.agent.deep_search import DeepSearch
from deep_search.agent.response_agent import ResponseAgent
from deep_search.utils.api import call_openai
from deep_search.utils.retry import retry

class HotpotQAEvaluator:
    def __init__(self, model_name="o1-mini-2024-09-12", data_path="data/hotpotqa/hotpotqa_100.jsonl", save_path="results", resume_path=None, data_dir="data", sample_num=100, tag="hotpotqa", 
            critic_times=1, mcp_server_name="search_only", ckpt_path=None, plan_strategy="fact_only", expand_num=2, search_steps=12, simu_steps=12, evaluator_configs=None, method_type="hierarchical", **kwargs):
        self.language = "en"
        self.model_name = model_name
        self.data_path = data_path
        self.critic_times = critic_times
        self.debug = True
        self.black_list = [] # os.listdir(self.black_list_dir)
        self.sample_num = sample_num
        self.data = self.load_data()
        data_name = self.data_path.split("/")[-1].split(".")[0]
        self.save_dir = os.path.join(save_path, f"{data_name}_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}")
        if resume_path:
            self.save_dir = resume_path
            self.results = self.load_results(resume_path)
        else:
            
            os.makedirs(self.save_dir, exist_ok=True)
            self.results = {}
            
            # Save initialization parameters to a config file
            self.save_config({
                'model_name': model_name,
                'data_path': data_path,
                'save_path': save_path,
                'sample_num': sample_num,
                'tag': tag,
                'critic_times': critic_times,
                'mcp_server_name': mcp_server_name,
                'ckpt_path': ckpt_path,
                'plan_strategy': plan_strategy,
                'expand_num': expand_num,
                'search_steps': search_steps,
                'simu_steps': simu_steps,
                'evaluator_configs': evaluator_configs,
                'method_type': method_type,
                **kwargs
            })
            
        self.correct = 0
        self.total = 0
        self.mcp_server_name = mcp_server_name
        self.ckpt_path = ckpt_path
        self.plan_strategy = plan_strategy
        self.expand_num = expand_num
        self.search_steps = search_steps
        self.simu_steps = simu_steps
        self.evaluator_configs = evaluator_configs
        self.method_type = method_type
        self.kwargs = kwargs
        self.total_time = 0

    def load_data(self):
        if self.data_path.endswith(".jsonl"):
            raw_data = pd.read_json(self.data_path,  lines=True)
            raw_data["_id"] = raw_data.index.astype(str)
        elif self.data_path.endswith(".json"):
            raw_data = pd.read_json(self.data_path)
            raw_data["_id"] = raw_data.index.astype(str)
        if "Question" in raw_data.columns and "problem" not in raw_data.columns:
            raw_data.rename(columns={"Question": "problem"}, inplace=True)
        return raw_data.to_dict('records')[:self.sample_num]

    async def get_answer(self, task_id, query, ground_truth):
        if f"{task_id}.json" in self.black_list:
            return False, False, [], "", ""
        try:
            dr = DeepSearch(model=self.model_name, task=query, max_steps=self.search_steps, simu_steps=self.simu_steps, obs_review_max_turn=self.critic_times,
                     mcp_server_name=self.mcp_server_name, plan_strategy=self.plan_strategy, expand_num=self.expand_num, evaluator_configs=self.evaluator_configs,
                     method_type=self.method_type, **self.kwargs)
            await dr.initialize()
            start_time = time.time()
            messages, other_info = await dr()
            response_agent = ResponseAgent(self.model_name)
            response = response_agent.response(query, messages)
            end_time = time.time()
            self.total_time += end_time - start_time
        except Exception as e:
            with open(os.path.join(self.black_list_dir, f"{task_id}.json"), "w") as f:
                json.dump({"query": query, "reason": str(e)}, f, indent=4, ensure_ascii=False)
            return False, False, [], "", ""
        finally:
            await dr.cleanup()
        return True, messages, response, "", other_info

    def _translate(self, query):
        if self.language == "zh":
            query = call_openai("gpt-4o-2024-11-20", [{"role": "user", 
                                                      "content": f"请将以下问题翻译成通顺的中文：{query}。如果你翻译人名、专有名词或技术术语时，请在括号中保留英文原文，例如：氯化镉(Cadmium Chloride)、乔治·卢卡斯(George Lucas)。不要连续出现括号前和括号中都是英文的情况。输出时只输出翻译后的文本，不要加任何修饰。"}])['content']
            assert "翻译" not in query, f"Translation failed: {query}"
            return query
        else:
            return query

    @retry(3,1)
    def judger(self, question: str, answer: str, standard_answer: str):
        system_prompt = f"""You are a professional evaluator. Please judge whether the user's answer is correct based on the standard answer. The standard answer is always correct, please use it as the reference.

##Question
{question}

##Standard Answer
{standard_answer}

##User Answer
{answer}

Based on the given question and corresponding standard answer, determine if the user's answer matches the standard answer (whether it correctly answers the question). Return in JSON format.
    {{
        "reason": "The reason why the user's answer is correct or incorrect given the current question and standard answer",
        "is_correct": 1 or 0
    }}
    """
        msg = [
            {"role": "user", "content": system_prompt},
        ]
        print(standard_answer)
        result = call_openai(model="gpt-4o-2024-11-20", messages=msg, response_format={"type": "json_object"})["content"]
        is_correct = json.loads(result)["is_correct"]
        if is_correct in [1, "1", True, "true"]:
            return True
        else:
            return False

    async def _evaluate_sample(self, sample):
        task_id = sample['_id']
        response_agent = ResponseAgent(self.model_name)
        if task_id in self.results:
            # response = response_agent.response(sample['problem'], self.results[task_id]['messages'])
            ground_truth = " / ".join(sample['answer']) if isinstance(sample['answer'], list) else sample['answer']
            # is_correct = self.judger(sample['problem'], self.results[task_id]['response'], ground_truth)
            # # self.results[task_id]['origin_response'] = self.results[task_id]['response']
            # # self.results[task_id]['response'] = response
            # self.results[task_id]['is_correct'] = is_correct
            # self.save_results(self.results[task_id])
            time.sleep(random.random() * 3)
            return self.results[task_id]
        query = self._translate(sample['problem'])
        ground_truth = " / ".join(sample['answer']) if isinstance(sample['answer'], list) else sample['answer']
        is_enough, messages, response, keypoints, other_info = await self.get_answer(task_id, query, ground_truth)
        if response is not None:
            is_correct = self.judger(query, response, ground_truth)
        else:
            is_correct = None
        result = {
            'task_id': task_id,
            'query': query,
            "response": response,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'is_enough': is_enough,
            'sample': sample,
            "keypoints": keypoints,
            "messages": messages,
            "other_info": other_info
        }
        self.save_results(result)
        print(f"Sample {task_id}")
        print(f"Query: {query}")
        print(f"Prediction: {response}")
        print(f"Ground truth: {ground_truth}")
        print(f"Correct: {is_correct}\n")
        return result

    def evaluate(self):
        start_time = time.time()
        if self.debug:
            for sample in tqdm(self.data, desc="Evaluating"):
                result = asyncio.run(self._evaluate_sample(sample))
                self.results[str(result['task_id'])] = result
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_tuple = {executor.submit(asyncio.run, self._evaluate_sample(sample)): sample for sample in self.data}
                for future in tqdm(concurrent.futures.as_completed(future_to_tuple), total=len(future_to_tuple), desc="Evaluating"):
                    sample = future_to_tuple[future]
                    try:
                        result = future.result()
                        self.results[str(result['task_id'])] = result
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        print(f"search error: {e}, sample id: {sample['_id']}")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        for result_id in self.results.keys():
            if f"{result_id}.json" not in os.listdir(self.save_dir):
                black_list_file = os.path.join(self.black_list_dir, f"{result_id}.json")
                with open(black_list_file, "w") as f:
                    json.dump(self.results[result_id], f, indent=4, ensure_ascii=False)

    def get_accuracy(self):
        correct = sum(result['is_correct'] for result in self.results.values())
        total = len(self.data)
        # import pdb; pdb.set_trace()
        if total == 0:
            return 0.0
        return correct / total * 100
    
    def save_results(self, result):
        save_file_path = os.path.join(self.save_dir, f"{result['sample']['_id']}.json")
        with open(save_file_path, 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    def load_results(self, resume_path):
        results = {}
        for file in os.listdir(resume_path):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(resume_path, file), 'r') as f:
                try:
                    results[file.split('.')[0]] = json.load(f)
                except Exception as e:
                    continue
                    print(f"Error loading {file}: {e}")
        return results

    def save_config(self, config):
        """Save the configuration parameters to a file."""
        config_path = os.path.join(self.save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=4, allow_unicode=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the HotpotQA model.")
    # parser.add_argument('--model_name', type=str, default="gpt-4o-2024-11-20", choices=["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "o1-mini-2024-09-12", "o3-mini-2025-01-31", "QwQ-32B"], help='Name of the model to evaluate.')
    # parser.add_argument('--data_path', type=str, default="data/hotpotqa/hotpotqa_100.jsonl", help='Path to the dataset.')
    # parser.add_argument('--save_path', type=str, default="results/hotpotqa_100", help='Path to save the evaluation results.')
    # parser.add_argument('--resume_path', type=str, default=None, help='Path to resume the evaluation.')
    # parser.add_argument('--tag', type=str, default="hotpotqa", help='Tag for the evaluation results.')
    # parser.add_argument('--critic_times', type=int, default=1, help='Number of times to criticize the model.')
    # parser.add_argument('--sample_num', type=int, default=100, help='Number of samples to evaluate.')
    # parser.add_argument('--plan_strategy', type=str, default="fact", choices=["fact", "planning", "plan_only"], help='Agent plan strategy.')
    # parser.add_argument('--expand_num', type=int, default=2, help='Number of expand nodes.')
    # parser.add_argument('--search_steps', type=int, default=12, help='Number of search steps.')
    # parser.add_argument('--simu_steps', type=int, default=12, help='Number of simulation steps.')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file.')
    args = parser.parse_args()
    
    default_config = {
        "model_name": "gpt-4o-2024-11-20",
        "data_path": "data/hotpotqa/hotpotqa_100.jsonl",
        "save_path": "results/hotpotqa_100",
        "resume_path": None,
        "tag": "hotpotqa",
        "critic_times": 1,
        "sample_num": 100,
        "plan_strategy": "fact",
        "expand_num": 2,
        "search_steps": 12,
        "simu_steps": 12,
        "evaluator_configs": None,
        "method_type": "hierarchical"
    }
    
    # If config file is provided, load it and update args
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    return config

if __name__ == "__main__":
    config = parse_args()
    evaluator = HotpotQAEvaluator(
        **config
    )
    evaluator.evaluate()
    print(f"Final accuracy: {evaluator.get_accuracy():.2f}%")
    # 写一段代码测试judger
    # judger = HotpotQAEvaluator(model_name="gpt-4o-mini-2024-07-18")
    # print(judger.judger("Veljko Čubrilović was involved in the assassination of which heir to the Austro-Hungarian throne?", "Archduke Franz Ferdinand of Austria.", "Franz Ferdinand Carl Ludwig Joseph Maria"))

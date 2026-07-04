import re
from deep_search.utils.api import call_openai
from deep_search.utils.retry import retry

class ResponseAgent:
    def __init__(self, model: str):
        self.model = model

    @retry(10, 1)
    def response(self, query: str, messages: list[dict]) -> str:
        context = "\n".join([f"--------{message['role']}--------\n{message['content']}" for message in messages if message['content'] != ""])
        sys_prompt = """You are a question answering AI assistant. I provide you a question and search history. You should answer the question based on the search history. Gather all the information from the search history and then answer the question in a few words or a sentence.
        """
        user_prompt = f"""## Search History
{context}

## Question
{query}

Now answer the question based on the search history in english and state the answer at the beginning of your response."""
        res = call_openai(model=self.model, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])["content"]
        # Extract final answer using regex
        # match = re.search(r"FINAL ANSWER:\s*(.*?)(?:\n|$)", res)
        # if match:
        #     res = match.group(1).strip()
        return res

class ResponseWithThinkingAgent:
    def __init__(self, model: str):
        self.model = model

    @retry(10, 1)
    def response(self, query: str, messages: list[dict]) -> str:
        context = "\n".join([f"--------{message['role']}--------\n{message['content']}" for message in messages if message['content'] != ""])
        sys_prompt = """You are a question answering AI assistant. I provide you a question and search history. You should answer the question based on the search history. Gather all the information from the search history, think step by step, and then answer the question in a few words or a sentence in following format "Final Answer: <answer>"."""
        user_prompt = f"""## Search History
{context}

## Question
{query}

Now answer the question based on the search history in english."""
        
        res = call_openai(model=self.model, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])["content"]
        # Extract final answer using regex
        # match = re.search(r"FINAL ANSWER:\s*(.*?)(?:\n|$)", res)
        # if match:
        #     res = match.group(1).strip()
        return res

if __name__ == "__main__":
    import json
    from deep_search.utils.judger import judger
    import os
    from tqdm import tqdm

    data_dir = os.getenv("RESPONSE_DATA_DIR", "results/example_run")
    tag = "with_thinking"
    save_dir = os.path.join("results", "test_response", tag, os.path.basename(data_dir))
    os.makedirs(save_dir, exist_ok=True)
    response_agent = ResponseWithThinkingAgent(model="gpt-4o-2024-11-20")
    total_correct = 0
    total_num = 0
    from concurrent.futures import ThreadPoolExecutor

    def process_file(file):
        if not file.endswith(".json"):
            return 0, 0
        with open(os.path.join(data_dir, file), "r") as f:
            if file in os.listdir(save_dir):
                data = json.load(open(os.path.join(save_dir, file), "r"))
                return data["is_correct"], 1
            data = json.load(f)
            question = data["query"]
            standard_answer = data["ground_truth"]
            response = response_agent.response(question, data["messages"])
            final_answer = response.split("Final Answer:")[1].strip()
            is_correct = judger(question, final_answer, standard_answer)
            data["is_correct"] = is_correct
            data["response"] = final_answer
            data["thinking"] = response
            print("question: ", question)
            print("response: ", response)
            print("final_answer: ", final_answer)
            print("standard_answer: ", standard_answer)
            print("is_correct: ", is_correct)
            with open(os.path.join(save_dir, file), "w") as f:
                json.dump(data, f, indent=4)
            return is_correct, 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_file, os.listdir(data_dir)), total=len(os.listdir(data_dir))))
    
    for is_correct, count in results:
        total_correct += is_correct
        total_num += count
    print("total_correct: ", total_correct)
    print("total_num: ", total_num)
    print("accuracy: ", total_correct / total_num)

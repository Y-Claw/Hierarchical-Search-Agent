import re
import os
import json
import concurrent
import time
import random
from datetime import datetime
from tqdm import tqdm
import argparse

from datasets import load_dataset
from deep_search.agent.deep_search import DeepSearch
from deep_search.agent.response_agent import ResponseAgent
from deep_search.utils.retry import retry

class GaiaEvaluator:
    
    def __init__(self, think_model="o1-mini-2024-09-12", toolcall_model="gpt-4o-2024-11-20", response_model="gpt-4o-2024-11-20", split="validation", level="2023_level2", modal='search-only', read_file="exclude", save_path="results", resume_path=None, data_root="data/gaia", concurrents=20, tag="no_tag"):
        self.think_model = think_model
        self.toolcall_model = toolcall_model
        self.response_model = response_model
        self.split = split
        self.level = level
        self.modal = modal
        self.read_file = read_file
        self.debug = os.getenv("DEBUG") == "True"
        self.data_root = os.path.join(data_root, self.split)
        self.data = self.load_data()
        self.save_dir = os.path.join(save_path, f"gaia_{self.think_model}_{self.toolcall_model}_{self.split}_{self.level}_{self.modal}_attachments_{self.read_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}")
        if resume_path:
            self.save_dir = resume_path
            self.results = self.load_results(resume_path)
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            self.results = {}
        self.concurrents = concurrents
        self.correct = 0
        self.total = 0

    @property
    def all_tools(self):
        return ['(Optional) Search engine', '(Optional) Web browser',
       'A Python IDE', 'A calculator', 'A calculator.',
       'A file interface', 'A search engine', 'A search engine.',
       'A speech-to-text audio processing tool', 'A speech-to-text tool',
       'A web browser', 'A web browser.', 'A word reversal tool / script',
       'Access to Wikipedia', 'Access to academic journal websites',
       'Access to the Internet Archive, web.archive.org',
       'Audio capability', 'Audio processing software',
       'Bablyonian cuniform -> arabic legend', 'Bass note data',
       'C++ compiler', 'CSV file access', 'Calculator',
       'Calculator (or use Excel)', 'Calculator or counting function',
       'Color recognition', 'Computer vision', 'Computer vision or OCR',
       'Counter', 'Excel', 'Excel file access', 'File handling',
       'GIF parsing tools', 'Google Maps', 'Google Translate access',
       'Graph interaction tools', 'Image processing tools',
       'Image recognition', 'Image recognition and processing tools',
       'Image recognition tools',
       'Image recognition tools (to identify and parse a figure with three axes)',
       'Image recognition/OCR', 'JSONLD file access', 'Markdown',
       'Microsoft Excel', 'Microsoft Excel / Google Sheets',
       'Natural language processor', 'OCR', 'PDF access', 'PDF reader',
       'PDF viewer', 'PowerPoint viewer', 'Python', 'Python compiler',
       "Rubik's cube model", 'Search Engine', 'Search engine',
       'Spreadsheet editor', 'Text Editor', 'Text processing/diff tool',
       'Tool to extract text from images', 'Unlambda compiler (optional)',
       'Video capability', 'Video parsing', 'Video processing software',
       'Video recognition tools', 'Web Browser', 'Web browser',
       'Wikipedia', 'Word document access', 'XLSX file access',
       'XML file access', 'YouTube', 'YouTube player', 'a calculator',
       'age recognition software', 'b Browser', 'calculator',
       'code/data analysis tools', 'computer algebra system',
       'google search', 'image recognition tools',
       'image recognition/OCR', 'image search tools', 'ne',
       'pdf reader/extracter', 'search engine', 'tools required',
       'video recognition tools', 'web browser']

    def invalid_tools(self, modal):
        if modal == 'search-only':
            invalid_tools = ['A file interface', 'A speech-to-text audio processing tool', 'A speech-to-text tool',
                            'Audio capability', 'Audio processing software', 'Color recognition', 'Computer vision', 
                            'Computer vision or OCR', 'GIF parsing tools', 'Graph interaction tools', 
                            'Image processing tools', 'Image recognition', 'Image recognition and processing tools', 
                            'Image recognition tools', 'Image recognition tools (to identify and parse a figure with three axes)',
                            'Image recognition/OCR', 'OCR', 'Tool to extract text from images', 'Video capability', 
                            'Video parsing', 'Video processing software', 'Video recognition tools', 
                            'YouTube', 'YouTube player', 'age recognition software', 'image recognition tools',
                            'image recognition/OCR', 'image search tools', 'ne', 'video recognition tools']
        else:
            invalid_tools = []
        return invalid_tools

    def have_invalid_tools(self, tools):
        if self.modal == 'search-only':
            return any(tool[2:].strip() in self.invalid_tools("search-only") for tool in tools)
        elif self.modal == 'multimodal-only':
            return not any(tool[2:].strip() in self.invalid_tools("search-only") for tool in tools)
        else:
            return False

    def filter_read_file(self, data):
        filtered_data = []
        if self.read_file == "exclude":
            for i in range(len(data)):
                if not data[i]['file_name']:
                    filtered_data.append(data[i])
        elif self.read_file == "include_only":
            for i in range(len(data)):
                if data[i]['file_name']:
                    filtered_data.append(data[i])
        else:
            filtered_data = data
        return filtered_data


    def load_data(self):
        raw_data = load_dataset('gaia-benchmark/GAIA', self.level, trust_remote_code=True)
        data = []
        tool_calls = []
        for i in range(len(raw_data[self.split])):
            tools = raw_data[self.split][i]['Annotator Metadata']['Tools'].split('\n')
            if self.have_invalid_tools(tools):
                continue
            tool_calls.extend([tool[2:].strip() for tool in tools])
            data.append(raw_data[self.split][i])
        data = self.filter_read_file(data)
        return data if not self.debug else data[:1]

    @retry(3, 1)
    def get_answer(self, query, attachments):
        dr = DeepSearch(think_model=self.think_model, toolcall_model=self.toolcall_model, agent_work_dir="downloads")
        dr.search(query, attachments)
        history = dr.get_search_info()
        response_agent = ResponseAgent(model=self.response_model)
        res = response_agent.response(query=query, context=history)

        return res, "", history

    def _evaluate_sample(self, sample):
        task_id = sample['task_id']
        query = sample['Question']
        ground_truth = sample['Final answer']
        attach_file_name = sample['file_name']
        if attach_file_name:
            attach_path = os.path.abspath(os.path.join(self.data_root, attach_file_name))
            attachments = {attach_file_name: attach_path}
        else:
            attachments = {}
        pred, outline, info = self.get_answer(query, attachments)
        is_correct = pred.strip().lower() == ground_truth.strip().lower()
        result = {
            'task_id': task_id,
            'query': query,
            'prediction': pred,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'sample': sample,
            "outline": outline,
            "info": info
        }
        self.save_results(result)
        print(f"Sample {task_id}")
        print(f"Prediction: {pred}")
        print(f"Ground truth: {ground_truth}")
        print(f"Correct: {is_correct}\n")
        return result

    def _evaluate(self, sample):
        task_id = sample['task_id']
        if task_id in self.results:
            result = self.results[task_id]
            time.sleep(random.randint(1, 3))
            result['task_id'] = task_id
            return result
        query = sample['Question']
        ground_truth = sample['Final answer']
        try:
            result = self._evaluate_sample(sample)
            return result
        except Exception as e:
            if self.debug:
                import pdb; pdb.set_trace()
            error_msg = f"Error processing sample {task_id}: {str(e)}"
            import traceback
            print(traceback.format_exc())
            print(error_msg)
            return {
                'task_id': task_id,
                'query': query,
                'prediction': error_msg,
                'ground_truth': ground_truth,
                'is_correct': False,
                'sample': sample,
                "outline": "",
                "info": ""
            }

    def evaluate(self):
        start_time = time.time()
        if self.debug:
            for i, sample in tqdm(enumerate(self.data), total=len(self.data), desc="Evaluating"):
                result = self._evaluate(sample)
                self.results[result['task_id']] = result
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrents) as executor:
                future_to_tuple = {executor.submit(self._evaluate, sample): sample for sample in self.data}
                for future in tqdm(concurrent.futures.as_completed(future_to_tuple), total=len(future_to_tuple), desc="Evaluating"):
                    sample = future_to_tuple[future]
                    try:
                        result = future.result()
                        self.results[result['task_id']] = result
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        print(f"search error: {e}, sample id: {sample['task_id']}")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    def get_accuracy(self):
        correct = sum(result['is_correct'] for result in self.results.values())
        total = len(self.data)
        if total == 0:
            return 0.0
        return correct / total * 100
    
    def save_results(self, result):
        save_file_path = os.path.join(self.save_dir, f"{result['sample']['task_id']}.json")
        with open(save_file_path, 'a') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    def load_results(self, resume_path):
        results = {}
        count = 0
        for data in self.data:
            task_id = data['task_id']
            json_path = os.path.join(resume_path, f"{task_id}.json")
            if os.path.exists(json_path):
                results[task_id] = json.load(open(json_path, "r"))
                count += 1
        # for file in os.listdir(resume_path):
        #     count += 1
        #     with open(os.path.join(resume_path, file), 'r') as f:
        #         results[file.split('.')[0]] = json.load(f)
        print(f"Loaded {count} results from {resume_path}")
        return results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Gaia model.")
    parser.add_argument('--think_model', type=str, default="gpt-4o-2024-11-20", choices=["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31"], help='Name of the model to evaluate.')
    parser.add_argument('--toolcall_model', type=str, default="gpt-4o-2024-11-20", choices=["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31"], help='Name of the model to evaluate.')
    parser.add_argument('--response_model', type=str, default="gpt-4o-2024-11-20", choices=["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "o1-mini-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31"], help='Name of the model to evaluate.')
    parser.add_argument('--split', type=str, default="validation", help='Dataset split to use for evaluation.')
    parser.add_argument('--level', type=str, default="2023_level2", help='Level of the dataset to use.')
    parser.add_argument('--modal', type=str, default='search-only', help='Modal to use for evaluation.')
    parser.add_argument('--read_file', type=str, default="exclude", help='Flag to indicate if files should be read.')
    parser.add_argument('--data_root', type=str, default="data/gaia", help="Path to save evaluation data")
    parser.add_argument('--save_path', type=str, default="results", help='Path to save the evaluation results.')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to resume the evaluation.')
    parser.add_argument('--concurrents', type=int, default=20, help='Number of concurrents to use.')
    parser.add_argument('--tag', type=str, default="no_tag", help='Flag to indicate if debug mode is on.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluator = GaiaEvaluator(
        think_model=args.think_model,
        toolcall_model=args.toolcall_model,
        response_model=args.response_model,
        split=args.split,
        level=args.level,
        modal=args.modal,
        read_file=args.read_file,
        save_path=args.save_path,
        resume_path=args.resume_path,
        data_root=args.data_root,
        concurrents=args.concurrents,
        tag=args.tag
    )
    evaluator.evaluate()
    print(f"Final accuracy: {evaluator.get_accuracy():.2f}%")

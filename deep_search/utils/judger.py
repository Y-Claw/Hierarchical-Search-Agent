import json
from deep_search.utils.api import call_openai
from deep_search.utils.retry import retry

@retry(3, 1)
def judger(question: str, answer: str, standard_answer: str):
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
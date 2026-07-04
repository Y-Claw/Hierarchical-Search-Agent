from deep_search.utils.api import call_openai, gpt_json_load

class AuthorizeEvaluator:

    def __init__(self, model: str, score_map: dict= {1: 1, 2: 0.75, 3: 0.5, 4: 0.25}):
        self.model = model
        self.score_map = score_map

    def evaluate(self, query: str, state: dict) -> float:
        system_message = """You are a helpful assistant that evaluates the credibility of information. You should evaluate whether the information is credible. There are 3 levels of credibility:
1. All information source is have high credibility, e.g. official website, official document, etc.
2. A part of information source is have high credibility (e.g. official website, official document, etc.), while some information source is have medium credibility (e.g. news website, etc.).
3. All information source is have medium credibility, e.g. news website, etc.
4. All information source is have low credibility, e.g. blog, forum, etc.

You should return the score in the following format, and do not include any other text:
{
    "reason": "The information source is have high credibility.",
    "level": 1
}
        """
        user_message = f"""
Query: {query}
Information: {state["observation"]}
"""
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        response = call_openai(self.model, message)["content"]
        response_json = gpt_json_load(response)
        return self.score_map[response_json["level"]]

from deep_search.utils.api import call_openai, gpt_json_load

class PotentialEvaluator:

    def __init__(self, model: str, score_map: dict = {1: 1, 2: 0.5, 3: 0}):
        self.model = model
        self.score_map = score_map

    def evaluate(self, query: str, state: dict) -> float:
        system_message = """You are a helpful assistant that evaluates potential of information for further investigation. You should evaluate whether the information have potential for further investigation. There are 3 levels of potential:
1. The information is very relevant to the query, further investigation have high possibility to get helpful information.
2. The information is relevant to the query, have potential for further investigation.
3. The information is not relevant to the query and have no potential for further investigation.

You should return the score in the following format, and do not include any other text:
{
    "reason": "The information is very relevant to the query, further investigation have high possibility to get helpful information.",
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

    def evaluate_all(self, query: str, states: list[dict]) -> list[float]:
        return [self.evaluate(query, state) for state in states]

from deep_search.utils.api import call_openai, gpt_json_load

class InformationEvaluator:

    def __init__(self, model: str, score_map: dict = {1: 1, 2: 0.5, 3: 0}):
        self.model = model
        self.score_map = score_map

    def evaluate(self, query: str, state: dict) -> float:
        system_message = """You are a helpful assistant that evaluates the quality of information. You should evaluate whether the information is relevant to the query. There are 3 levels of relevance:
1. The information is directly relevant to the query and very helpful for answering the query.
2. The information is relevant to the query, providing some useful details that contribute to answering the query or aiding in further investigation.
3. The information is not relevant to the query.

You should return the score in the following format, and do not include any other text:
{
    "reason": "The information is directly relevant to the query and very helpful for answering the query.",
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
    
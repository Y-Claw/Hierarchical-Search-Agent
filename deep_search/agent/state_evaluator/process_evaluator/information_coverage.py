from deep_search.utils.api import call_openai, gpt_json_load

class InformationCoverageEvaluator:

    def __init__(self, model: str, score_map: dict = {1: 1, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0}, evaluation_mode: str = "single"):
        self.model = model
        self.score_map = score_map
        self.evaluation_mode = evaluation_mode

    def evaluate(self, query: str, messages: list[dict]) -> float:
        system_message = """You are a helpful assistant that evaluates the quality of information. You should evaluate whether search trajectory have thorough coverage the necessary information of the query. There are 5 levels of density:
1. Excellent coverage; the search trajectory covers all the necessary information of the query and gather as much as possible information from multiple sources.
2. Good coverage; the search trajectory covers all the necessary information of the query.
3. Moderate coverage; the search trajectory covers most of the necessary information of the query.
4. Limited coverage; the search trajectory covers some necessary information of the query but still lacks comprehensive coverage.
5. Insufficient coverage; the search trajectory does not cover the necessary information of the query.

You should return the score in the following format, and do not include any other text:
{
    "reason": "The search trajectory is insufficiently comprehensive, lacking additional insights that could enhance the understanding of the initial query.",
    "level": 1
}
        """
        context = "\n".join([f"--------{message['role']}--------\n{message['content']}" for message in messages if message['content'] != ""])
        user_message = f"""
Search History:
{context}

Query: {query}
"""
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        response = call_openai(self.model, message)["content"]
        response_json = gpt_json_load(response)
        return self.score_map[response_json["level"]]

    def evaluate_all(self, query: str, messages: list[dict]) -> list[float]:
        if self.evaluation_mode == "single":
            return [self.evaluate(query, messages)]
        elif self.evaluation_mode == "all":
            return [self.evaluate(query, messages) for messages in messages]
        else:
            raise ValueError(f"Invalid evaluation mode: {self.evaluation_mode}")

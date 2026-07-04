from deep_search.utils.api import call_openai, gpt_json_load

class InformationRelevanceEvaluator:

    def __init__(self, model: str, score_map: dict = {1: 1, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0}, evaluation_mode: str = "single"):
        self.model = model
        self.score_map = score_map
        self.evaluation_mode = evaluation_mode

    def evaluate(self, query: str, messages: list[dict]) -> float:
        system_message = """You are a helpful assistant that evaluates the quality of information. You should evaluate whether the information gathered in each step of the search trajectory is relevant to the query. There are 5 levels of relevance:
1. Excellent relevance; The search trajectory is fully relevant to the query, gathered all the key information to answer the query and gather as much as possible information from multiple sources.
2. Good relevance; The search trajectory is fully relevant to the query and gathered key information to answer the query.
3. Moderate relevance; Most of the search steps are relevant to the query, but the search trajectory may stick to some unimportant information.
4. Limited relevance; Some search steps are relevant to the query, but the search trajectory is deviated from the query.
5. Insufficient relevance; Most of the search steps are not relevant to the query, especially at the beginning of the search trajectory.

You should return the score in the following format, and do not include any other text:
{
    "reason": "The search trajectory is fully relevant to the query and gathered all the key information from multiple sources.",
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

    def evaluate_pair(self, query, msg_1, msg_2):
        system_message = """You are a helpful assistant tasked with evaluating the quality of information. You need to compare two search histories to determine which one is more relevant to the query. There are 5 levels of relevance:
1. Excellent relevance; The search history is fully relevant to the query, gathering all the key information needed to answer the query and collecting as much information as possible from multiple sources.
2. Good relevance; The search history is fully relevant to the query and has gathered key information to answer the query.
3. Moderate relevance; Most of the search steps are relevant to the query, but the search history may include some unimportant information.
4. Limited relevance; Some search steps are relevant to the query, but the search history deviates from the query.
5. Insufficient relevance; Most of the search steps are not relevant to the query, especially at the beginning of the search history.
Please evaluate which search history is more relevant to the query. You should think step by step and provide your reasoning. return the result in the following format:
{
    "reason": "The search history 1 contains ..., but the search history 2 does not contain ... . So the search history 1 is more relevant to the query.",
    "better_history": 1 / 2
}

"""
        context_1 = "\n".join([f"--------{message['role']}--------\n{message['content']}" for message in msg_1 if message['content'] != ""])
        context_2 = "\n".join([f"--------{message['role']}--------\n{message['content']}" for message in msg_2 if message['content'] != ""])

        user_message = f"""
Query: {query}

Search History 1:
{context_1}

Search History 2:
{context_2}

Now, please evaluate which search history is more relevant to the query.
"""
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        response = call_openai(self.model, message)["content"]
        response_json = gpt_json_load(response)
        return response_json["better_history"]

    def swiss_round(self, query: str, messages_list: list[list[dict]]) -> list[float]:
        # 使用evaluate_pair进行瑞士轮评分，并返回最终评估的分数
        scores = [0] * len(messages_list)
        import concurrent.futures

        def evaluate_pair_concurrent(i, j):
            better_history = self.evaluate_pair(query, messages_list[i], messages_list[j])
            return i, j, better_history

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_pair_concurrent, i, j) for i in range(len(messages_list)) for j in range(i + 1, len(messages_list))]
            for future in concurrent.futures.as_completed(futures):
                i, j, better_history = future.result()
                if better_history == 1:
                    scores[i] += 1
                elif better_history == 2:
                    scores[j] += 1
        return [score / (len(messages_list) - 1) for score in scores]

    def evaluate_all(self, query: str, messages_list: list[list[dict]]) -> list[float]:
        if self.evaluation_mode == "single":
            return [self.evaluate(query, messages) for messages in messages_list]
        elif self.evaluation_mode == "swiss":
            # 使用evaluate_pair进行瑞士轮评分，并返回最终评估的分数
            return self.swiss_round(query, messages_list)
        else:
            raise ValueError(f"Invalid evaluation mode: {self.evaluation_mode}")

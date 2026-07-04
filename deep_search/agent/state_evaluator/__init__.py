from deep_search.agent.state_evaluator.process_evaluator.information_coverage import InformationCoverageEvaluator
from deep_search.agent.state_evaluator.process_evaluator.information_relevance import InformationRelevanceEvaluator
from deep_search.agent.state_evaluator.process_evaluator.information_richness import InformationRichnessEvaluator
from deep_search.agent.state_evaluator.information_evaluator import InformationEvaluator
from deep_search.agent.state_evaluator.authorize_evaluator import AuthorizeEvaluator
from deep_search.agent.state_evaluator.potential_evaluator import PotentialEvaluator

EVALUATORS = {
    "coverage_evaluator": InformationCoverageEvaluator,
    "relevance_evaluator": InformationRelevanceEvaluator,
    "richness_evaluator": InformationRichnessEvaluator,
    "information_evaluator": InformationEvaluator,
    "authorize_evaluator": AuthorizeEvaluator,
    "potential_evaluator": PotentialEvaluator,
}
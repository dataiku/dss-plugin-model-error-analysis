import logging
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer
from dku_error_analysis_utils import ErrorAnalyzerConstants

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Error Analysis Plugin %(levelname)s - %(message)s")

class TreeHandler(object):
    DEFAULT_MAX_NR_FEATURES = 5
    def __init__(self):
        self.initialize()
        self.analyzer = None

    def initialize(self):
        self.selected_feature_ids = set()
        self.already_fetched_locally = set()
        self.already_fetched_globally = set()
        self.current_node_id = None

    @property
    def tree(self):
        if self.analyzer is not None:
            return self.analyzer.tree

    def train_tree(self):
        """
        Fit the Decision Tree and parse it so it can be viewed in the webapp

        :return: The accuracy of the original model, computed on the part of the test set used to train the Error Analyzer Tree
        """
        self.analyzer.fit()
        self.analyzer.parse_tree()

        self.selected_feature_ids = set(range(min(len(self.tree.ranked_features), self.DEFAULT_MAX_NR_FEATURES)))

        summary = self.analyzer.evaluate(output_format='dict')
        confidence_decision = summary[ErrorAnalyzerConstants.CONFIDENCE_DECISION]
        if not confidence_decision:
            # TODO: add message in UI
            LOGGER.warning("Warning: the built Error Analyzer Tree might not be representative of the original model performances.")

        return summary[ErrorAnalyzerConstants.PRIMARY_MODEL_TRUE_ACCURACY] # TODO: compute proper value

    def set_error_analyzer(self, original_model_handler):
        self.analyzer = DkuErrorAnalyzer(original_model_handler)

    def set_current_node_id(self, node_id):
        self.current_node_id = node_id
        self.already_fetched_locally = set()

    def set_selected_feature_ids(self, feature_ids):
        new_ids = feature_ids - self.selected_feature_ids
        if self.current_node_id is not None:
            self.already_fetched_locally |= new_ids
        self.selected_feature_ids = feature_ids

    def get_stats_node(self, node_id):
        self.set_current_node_id(node_id)
        return self._get_stats_node(node_id, self.already_fetched_locally)

    def get_stats_root(self):
        return self._get_stats_node(0, self.already_fetched_globally)

    def _get_stats_node(self, node_id, excluded_id_set):
        stats = {}
        for idx in self.selected_feature_ids - excluded_id_set:
            feature_name = self.tree.ranked_features[idx]["name"]
            stats[feature_name] = self.tree.get_stats(node_id, feature_name)
            excluded_id_set.add(idx)
        return stats

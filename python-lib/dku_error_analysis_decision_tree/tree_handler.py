import logging
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer
from dku_error_analysis_utils import ErrorAnalyzerConstants

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Error Analysis Plugin %(levelname)s - %(message)s")

class TreeHandler(object):
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

    def train_mpp(self):
        """
        Fit the Decision Tree and parse it so it can be viewed in the webapp

        :return: The accuracy of the original model, computed on the part of the test set used to train the MPP 
        """
        self.analyzer.fit()
        self.analyzer.parse_tree()

        summary = self.analyzer.mpp_summary(output_dict=True)
        confidence_decision = summary[ErrorAnalyzerConstants.CONFIDENCE_DECISION]
        if not confidence_decision:
            # TODO: add message in UI
            LOGGER.warning("Warning: the built MPP might not be representative of the original model performances.")

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
        return new_ids - self.already_fetched_globally

    def get_stats_node(self, node_id):
        self.set_current_node_id(node_id)
        stats = {}
        for idx in self.selected_feature_ids:
            if idx not in self.already_fetched_locally:
                feature_name = self.tree.ranked_features[idx]["name"]
                stats[feature_name] = self.tree.get_stats(node_id, feature_name)
                self.already_fetched_locally.add(idx)
        return stats

    def get_stats_root(self, global_data_to_fetch):
        stats = {}
        for idx in global_data_to_fetch:
            feature_name = self.tree.ranked_features[idx]["name"]
            stats[feature_name] = self.tree.get_stats(0, feature_name)
            self.already_fetched_globally.add(idx)
        return stats

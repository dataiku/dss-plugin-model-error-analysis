import traceback, logging, json
from flask import jsonify, request

import dataiku

from dataiku.customwebapp import get_webapp_config
from dataiku.core.dkujson import DKUJSONEncoder

from dku_error_analysis_model_parser.model_metadata import get_model_handler
from dku_error_analysis_model_parser.model_accessor import ModelAccessor
from dku_error_analysis_mpp.dku_error_analyzer import DkuErrorAnalyzer
from dku_error_analysis_utils import ErrorAnalyzerConstants

app.json_encoder = DKUJSONEncoder

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Error Analysis Plugin %(levelname)s - %(message)s")

# initialization of the backend
MODEL_ID = get_webapp_config()["modelId"]
VERSION_ID = get_webapp_config()["versionId"]

class TreeHandler(object):
    def __init__(self):
        self.set_tree(None)

    def set_tree(self, tree):
        self.selected_feature_ids = set()
        self.already_fetched_locally = set()
        self.already_fetched_globally = set()
        self.current_node_id = None
        self.tree = tree

    def set_current_node_id(self, node_id):
        self.current_node_id = node_id
        self.already_fetched_locally = set()

    def set_selected_feature_ids(self, feature_ids):
        new_ids = feature_ids - self.selected_feature_ids
        if self.current_node_id is not None:
            self.already_fetched_locally |= new_ids
        self.selected_feature_ids = feature_ids
        return new_ids - self.already_fetched_globally

handler = TreeHandler()

def check_confidence(summary):
    confidence_decision = summary[ErrorAnalyzerConstants.CONFIDENCE_DECISION]

    if not confidence_decision:
        # TODO: add message in UI (ch49209)
        LOGGER.warning("Warning: the built MPP might not be representative of the primary model performances.")

def get_error_analyzer(model_handler):
    model_accessor = ModelAccessor(model_handler)
    dku_error_analyzer = DkuErrorAnalyzer(model_accessor)

    dku_error_analyzer.fit()
    dku_error_analyzer.parse_tree()
    return dku_error_analyzer

@app.route("/load", methods=["GET"])
def load():
    try:
        model_handler = get_model_handler(dataiku.Model(MODEL_ID), VERSION_ID)
        analyzer = get_error_analyzer(model_handler)
        summary = analyzer.mpp_summary(output_dict=True)
        check_confidence(summary)
        tree = analyzer.tree
        handler.set_tree(tree)

        return jsonify(nodes=tree.jsonify_nodes(),
            rankedFeatures=tree.ranked_features,
            estimatedAccuracy=summary[ErrorAnalyzerConstants.PRIMARY_MODEL_PREDICTED_ACCURACY],
            actualAccuracy=summary[ErrorAnalyzerConstants.PRIMARY_MODEL_TRUE_ACCURACY])
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/select-node/<int:node_id>")
def get_stats_node(node_id):
    try:
        handler.set_current_node_id(node_id)
        result = {}
        for idx in handler.selected_feature_ids:
            if idx not in handler.already_fetched_locally:
                feature_name = handler.tree.ranked_features[idx]["name"]
                result[feature_name] = handler.tree.get_stats(node_id, feature_name)
                handler.already_fetched_locally.add(idx)
        return jsonify(result)
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/select-features", methods=["POST"])
def select_features():
    try:
        feature_ids = set(json.loads(request.data)["feature_ids"])
        global_data_to_fetch = handler.set_selected_feature_ids(feature_ids)
        result = {}
        for idx in global_data_to_fetch:
            feature_name = handler.tree.ranked_features[idx]["name"]
            result[feature_name] = handler.tree.get_stats(0, feature_name)
            handler.already_fetched_globally.add(idx)
        return jsonify(result)
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

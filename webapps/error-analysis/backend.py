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
        self.features = []
        self.selected_features = []

    def set_tree(self, tree):
        self.tree = tree

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
        tree = handler.tree
        result = {}
        for feature in tree.ranked_features[:ErrorAnalyzerConstants.TOP_K_FEATURES]:
            feature_name = feature["name"]
            result[feature_name] = tree.get_stats(node_id, feature_name)
        return jsonify(result)
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/select-features", methods=["POST"])
def select_features():
    try:
        tree = handler.tree
        features = json.loads(request.data)["features"]
        pass
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

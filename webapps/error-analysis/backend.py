import traceback
import logging
from flask import jsonify
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

TREE = []

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
        TREE.append(tree)

        return jsonify(nodes=tree.jsonify_nodes(),
            target_values=tree.target_values,
            features=tree.features,
            rankedFeatures=tree.ranked_features[:ErrorAnalyzerConstants.TOP_K_FEATURES],
            estimatedAccuracy=summary[ErrorAnalyzerConstants.PRIMARY_MODEL_PREDICTED_ACCURACY],
            actualAccuracy=summary[ErrorAnalyzerConstants.PRIMARY_MODEL_TRUE_ACCURACY])
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/select-node/<int:node_id>")
def get_stats_node(node_id):
    try:
        tree = TREE[0]
        result = {}
        for feat in tree.ranked_features[:ErrorAnalyzerConstants.TOP_K_FEATURES]:
            result[feat] = tree.get_stats(node_id, feat)
        return jsonify(result)
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

import traceback, logging, json
from flask import jsonify, request

import dataiku

from dataiku.customwebapp import get_webapp_config
from dataiku.core.dkujson import DKUJSONEncoder

from dku_error_analysis_decision_tree.tree_handler import TreeHandler
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

handler = TreeHandler()

def check_confidence(summary):
    confidence_decision = summary[ErrorAnalyzerConstants.CONFIDENCE_DECISION]

    if not confidence_decision:
        # TODO: add message in UI (ch49209)
        LOGGER.warning("Warning: the built MPP might not be representative of the primary model performances.")

def get_error_analyzer(model_accessor):
    dku_error_analyzer = DkuErrorAnalyzer(model_accessor)

    dku_error_analyzer.fit()
    dku_error_analyzer.parse_tree()
    return dku_error_analyzer

@app.route("/original-model-info", methods=["GET"])
def get_original_model_info():
    try:
        model = dataiku.Model(MODEL_ID)
        model_accessor = ModelAccessor(get_model_handler(model, VERSION_ID))
        handler.set_original_model_accessor(model_accessor)

        return jsonify(modelName=model.get_name(), isRegression=model_accessor.is_regression())
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/load", methods=["GET"])
def load():
    try:
        if handler.original_model_accessor is None: # TODO
            model = dataiku.Model(MODEL_ID)
            model_accessor = ModelAccessor(get_model_handler(model, VERSION_ID))
            handler.set_original_model_accessor(model_accessor)
        analyzer = get_error_analyzer(handler.original_model_accessor)
        summary = analyzer.mpp_summary(output_dict=True)
        check_confidence(summary)
        handler.set_tree(analyzer.tree)

        return jsonify(nodes=handler.tree.jsonify_nodes(),
            rankedFeatures=handler.tree.ranked_features,
            actualAccuracy=summary[ErrorAnalyzerConstants.PRIMARY_MODEL_TRUE_ACCURACY])
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/select-node/<int:node_id>")
def get_stats_node(node_id):
    try:
        return jsonify(handler.get_stats_node(node_id))
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/global-chart-data")
def get_global_char_data():
    try:
        return jsonify(handler.get_stats_root())
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/select-features", methods=["POST"])
def select_features():
    try:
        feature_ids = set(json.loads(request.data)["feature_ids"])
        handler.set_selected_feature_ids(feature_ids)
        return "OK"
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

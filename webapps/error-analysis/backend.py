import traceback, logging, json
from flask import jsonify, request

import dataiku

from dataiku.customwebapp import get_webapp_config
from dataiku.core.dkujson import DKUJSONEncoder

from dku_error_analysis_model_parser.model_handler_utils import get_model_handler
from dku_error_analysis_decision_tree.tree_handler import TreeHandler

app.json_encoder = DKUJSONEncoder

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Error Analysis Plugin %(levelname)s - %(message)s")

# initialization of the backend
MODEL = dataiku.Model(get_webapp_config()["modelId"])
VERSION_ID = get_webapp_config().get("versionId")

handler = TreeHandler()

@app.route("/original-model-info", methods=["GET"])
def get_original_model_info():
    try:
        original_model_handler = get_model_handler(MODEL, VERSION_ID)
        handler.set_error_analyzer(original_model_handler)

        return jsonify(modelName=MODEL.get_name(),
            isRegression='REGRESSION' in original_model_handler.get_prediction_type())
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/load", methods=["GET"]) # Should always be called after a first call to /original-model-info
def load():
    try:
        handler.initialize()
        accuracy = handler.train_mpp()

        return jsonify(nodes=handler.tree.jsonify_nodes(),
            rankedFeatures=handler.tree.ranked_features,
            actualAccuracy=accuracy)
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

@app.route("/select-features", methods=["POST"])
def select_features():
    try:
        feature_ids = set(json.loads(request.data)["feature_ids"])
        global_data_to_fetch = handler.set_selected_feature_ids(feature_ids)
        return jsonify(handler.get_stats_root(global_data_to_fetch))
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

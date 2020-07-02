import traceback
import json
import logging
from flask import jsonify
import dataiku

from dataiku.customwebapp import get_webapp_config
from dataiku.core.dkujson import DKUJSONEncoder

from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_utils import safe_str
from dku_error_analysis_mpp.model_metadata import get_model_handler
from dku_error_analysis_mpp.error_analyzer import ErrorAnalyzer
from dku_error_analysis_mpp.model_accessor import ModelAccessor


app.json_encoder = DKUJSONEncoder

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Error Analysis Plugin %(levelname)s - %(message)s")

# initialization of the backend
MODEL_ID = get_webapp_config()["modelId"]
VERSION_ID = get_webapp_config()["versionId"]

TREE = []

def get_error_dt(model_handler):
    model_accessor = ModelAccessor(model_handler)
    error_analyzer = ErrorAnalyzer(model_accessor)

    error_analyzer.fit()
    tree = error_analyzer.tree

    return tree

@app.route("/load", methods=["GET"])
def load():
    try:
        model_handler = get_model_handler(dataiku.Model(MODEL_ID), VERSION_ID)
        tree = get_error_dt(model_handler)
        TREE.append(tree)
        return jsonify(nodes=tree.jsonify_nodes(), target_values=tree.target_values, features=tree.features, rankedFeatures=tree.ranked_features)
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/select-node/<int:node_id>")
def get_stats_node(node_id):
    try:
        tree = TREE[0]
        result = {}
        for feat in tree.ranked_features:
            result[feat] = tree.get_stats(node_id, feat)
        return jsonify(result)
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

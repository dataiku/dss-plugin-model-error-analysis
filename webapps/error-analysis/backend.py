import traceback
import json
import logging
from flask import jsonify
import dataiku

from dataiku.customwebapp import get_webapp_config
from dataiku.core.dkujson import DKUJSONEncoder

from dku_error_analysis_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_utils import safe_str
from dku_error_analysis_mpp.mpp_build import get_error_dt, rank_features_by_error_correlation
from dku_error_analysis_mpp.model_metadata import get_model_handler

app.json_encoder = DKUJSONEncoder

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Error Analysis Plugin %(levelname)s - %(message)s")

# initialization of the backend
MODEL_ID = get_webapp_config()["modelId"]
VERSION_ID = get_webapp_config()["versionId"]

TREE = []

@app.route("/load", methods=["GET"])
def load():
    try:
        model_handler = get_model_handler(dataiku.Model(MODEL_ID), VERSION_ID)
        clf, test_df, transformed_df, features = get_error_dt(model_handler)
        tree_parser = TreeParser(model_handler, clf.tree_)
        tree_parser.create_preprocessed_feature_mapping()
        ranked_features = rank_features_by_error_correlation(clf, features, tree_parser)
        tree = tree_parser.build_tree(test_df, ranked_features)
        tree_parser.build_all_nodes(tree, features, transformed_df)
        TREE.append(tree)
        return jsonify(nodes=tree.jsonify_nodes(), target_values=tree.target_values, features=tree.features, rankedFeatures=ranked_features)
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

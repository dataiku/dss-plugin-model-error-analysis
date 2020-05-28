import traceback
import json
import logging
from flask import jsonify
import dataiku

from dataiku.customwebapp import get_webapp_config
from dataiku.core.dkujson import DKUJSONEncoder
from dataiku.doctor.preprocessing.dataframe_preprocessing import RescalingProcessor2 as rescaler

from dku_error_tree_parsing.tree_parser import TreeParser
from dku_error_analysis_utils.compatibility import safe_str
from dku_error_analysis_mpp.mpp_build import get_error_dt, rank_features_by_error_correlation
from dku_error_analysis_mpp.model_metadata import get_model_handler
from dku_error_tree_parsing.depreprocessor import descale_numerical_thresholds

app.json_encoder = DKUJSONEncoder

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="MEA %(levelname)s - %(message)s")

# initialization of the backend
MODEL_ID = get_webapp_config()["modelId"]
VERSION_ID = get_webapp_config()["versionId"]

TREE = []

@app.route("/load", methods=["GET"])
def load():
    try:
        model_handler = get_model_handler(dataiku.Model(MODEL_ID), VERSION_ID)
        clf, test_df, features = get_error_dt(model_handler)
        rescalers = list(filter(lambda u: isinstance(u, rescaler), model_handler.get_pipeline().steps))
        thresholds = descale_numerical_thresholds(clf.tree_, features, rescalers, False)

        tree_parser = TreeParser(model_handler, thresholds, clf.tree_)
        ranked_features = rank_features_by_error_correlation(clf, features)
        tree = tree_parser.build_tree(test_df, ranked_features)
        tree_parser.build_all_nodes(clf.tree_, tree, thresholds, features)
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

import traceback, logging, json
from flask import jsonify, request

from dataiku import api_client, Model
from dataiku.customwebapp import get_webapp_config
from dataiku.core.dkujson import DKUJSONEncoder
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

from dataikuapi.dss.ml import DSSMLTask

from dku_error_analysis_decision_tree.tree_handler import TreeHandler

app.json_encoder = DKUJSONEncoder

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Error Analysis Plugin %(levelname)s - %(message)s")

# initialization of the backend
handler = TreeHandler()

@app.route("/original-model-info", methods=["GET"])
def get_original_model_info():
    try:
        fmi = get_webapp_config().get("trainedModelFullModelId")
        if fmi is None:
            model_id, version_id = get_webapp_config()["modelId"], get_webapp_config().get("versionId")
            model = Model(model_id)
            name = model.get_name()
            fmi = "S-{project_key}-{model_id}-{version_id}".format(project_key=model.project_key, model_id=model_id, version_id=version_id)
        else:
            name = DSSMLTask.from_full_model_id(api_client(), fmi).get_trained_model_snippet(fmi).get("userMeta", {}).get("name", fmi)
        original_model_handler = PredictionModelInformationHandler.from_full_model_id(fmi)
        handler.set_error_analyzer(original_model_handler)
        return jsonify(modelName=name,
            isRegression='REGRESSION' in original_model_handler.get_prediction_type())
    except:
        LOGGER.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route("/load", methods=["GET"]) # Should always be called after a first call to /original-model-info
def load():
    try:
        handler.initialize()
        accuracy = handler.train_tree()

        return jsonify(nodes=handler.tree.jsonify_nodes(),
            rankedFeatures=handler.tree.ranked_features,
            epsilon=handler.analyzer.epsilon,
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

from math import isnan
from dku_error_analysis_utils import safe_str
from mealy import ErrorAnalyzerConstants

class Node(object):
    """
    A node of a decision tree

    ATTRIBUTES
    id: positive integer, the id of the node

    parent_id: positive integer, the id of the parent node (worth -1 only for the root)

    children_ids: list of positive integers, the ids of the children nodes

    feature: string, the name of the feature of the split that has created the node

    probabilities: list of [value of target, probability of this value]

    prediction: string, name of the class of highest probability

    samples: positive integer, number of samples when applying the decision rules of the current node

    global_error: float

    local_error: array size two of format [local error, raw count of bad predictions in the node ]

    """

    class TYPES:
        NUM = "num"
        CAT = "cat"

    def __init__(self, node_id, parent_id, feature=None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.children_ids = []
        self.feature = feature
        self.probabilities = None
        self.prediction = None
        self.samples = None
        self.global_error = None
        self.local_error = None

    @property
    def id(self):
        return self.node_id

    def set_node_info(self, samples, total_samples, probabilities, prediction, global_error):
        self.samples = [samples, 100.0 * samples / total_samples]
        self.probabilities = []
        for class_name, class_samples in probabilities:
            self.probabilities.append([class_name, class_samples/samples, class_samples])
        self.prediction = prediction
        self.global_error = global_error

        if self.prediction and self.probabilities[0][0] == ErrorAnalyzerConstants.WRONG_PREDICTION:
            self.local_error = self.probabilities[0][1:3]
        else:
            self.local_error = self.probabilities[1][1:3]

    def get_type(self):
        raise NotImplementedError

    def jsonify(self):
        return dict(self.__dict__)

    def print_decision_rule(self):
        raise NotImplementedError

    def to_dot_string(self):
        dot_str = '{0} [label="node #{0}\n'.format(self.id)
        if self.parent_id >= 0:
            dot_str += self.print_decision_rule() + "\n"

        if self.local_error[0] >= ErrorAnalyzerConstants.GRAPH_MIN_LOCAL_ERROR_OPAQUE:
            alpha = 1.0
        else:
            alpha = self.local_error[0]

        node_class = ErrorAnalyzerConstants.CORRECT_PREDICTION if self.global_error == 0 else ErrorAnalyzerConstants.WRONG_PREDICTION
        class_color = ErrorAnalyzerConstants.ERROR_TREE_COLORS[node_class].strip('#')
        class_color_rgb = tuple(int(class_color[i:i + 2], 16) for i in (0, 2, 4))
        # compute the color as alpha against white
        color_rgb = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in class_color_rgb]
        color = '#{:02x}{:02x}{:02x}'.format(color_rgb[0], color_rgb[1], color_rgb[2])

        dot_str += 'samples = {:.3f}%\n'.format(self.samples[1])
        dot_str += 'local error = {:.3f}%\n'.format(100.*self.local_error[0])
        dot_str += 'fraction of total error = {:.3f}%\n'.format(100. * self.global_error)
        dot_str += '", fillcolor="{}"] ;'.format(color)
        return dot_str


class CategoricalNode(Node):
    def __init__(self, node_id, parent_id, feature, values, others=False):
        if values is None:
            raise ValueError()
        self.values = values
        self.others = others
        super(CategoricalNode, self).__init__(node_id, parent_id, feature)

    def get_type(self):
        return Node.TYPES.CAT

    def apply_filter(self, df):
        if self.others:
            return df[~df[self.feature].isin(self.values)]
        return df[df[self.feature].isin(self.values)]

    def jsonify(self):
        jsonified_dict = super(CategoricalNode, self).jsonify()
        first_value = jsonified_dict["values"][0]
        try:
            if isnan(first_value):
                jsonified_dict["values"] = ["No values"]
        except:
            pass
        return jsonified_dict

    def print_decision_rule(self):
        single_value = len(self.values) == 1
        if single_value:
            return self.feature + ' is ' + ( 'not ' if self.others else '') + safe_str(self.values[0])
        return self.feature + ( ' not ' if self.others else '') + ' in ['  + u', '.join(self.values) + "]"


class NumericalNode(Node):
    def __init__(self, node_id, parent_id, feature, beginning=None, end=None):
        if beginning is None and end is None:
            raise ValueError("A numerical node needs either an upper or lower bound")
        self.beginning = beginning
        self.end = end
        super(NumericalNode, self).__init__(node_id, parent_id, feature)

    def get_type(self):
        return Node.TYPES.NUM

    def apply_filter(self, df, mean=None):
        if self.beginning is not None:
            df = df[df[self.feature].gt(self.beginning, fill_value=mean)]
        if self.end is not None:
            df = df[df[self.feature].le(self.end, fill_value=mean)]
        return df

    def jsonify(self):
        jsonified_dict = super(NumericalNode, self).jsonify()
        if self.beginning is None:
            jsonified_dict.pop("beginning")
        elif self.end is None:
            jsonified_dict.pop("end")
        return jsonified_dict

    def print_decision_rule(self):
        decision_rule = self.feature
        if self.beginning:
            decision_rule = '{:.2f} < {}'.format(self.beginning, self.feature)            
        if self.end:
            decision_rule = '{} <= {:.2f}'.format(decision_rule, self.end)
        return decision_rule

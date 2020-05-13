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

    """

    class TYPES:
        NUM = "num"
        CAT = "cat"

    def __init__(self, node_id, parent_id, feature=None):
        self.id = node_id
        self.parent_id = parent_id
        self.children_ids = []
        self.feature = feature
        self.probabilities = None
        self.prediction = None
        self.samples = None
        self.global_error = None

    def set_node_info(self, samples, total_samples, probabilities, prediction, error):
        self.samples = [samples, 100.0 * samples / total_samples]
        self.probabilities = probabilities
        self.prediction = prediction
        self.global_error = error

    def get_type(self):
        raise NotImplementedError

    def jsonify(self):
        jsonified_node = dict(self.__dict__)
        return jsonified_node


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


class NumericalNode(Node):
    def __init__(self, node_id, parent_id, feature, beginning=None, end=None):
        if beginning is None and end is None:
            raise ValueError("A numerical node needs either an upper or lower bound")
        self.beginning = beginning
        self.end = end
        super(NumericalNode, self).__init__(node_id, parent_id, feature)

    def get_type(self):
        return Node.TYPES.NUM

    def apply_filter(self, df, mean):
        if self.beginning is not None:
            df = df[df[self.feature].ge(self.beginning, fill_value=mean)]
        if self.end is not None:
            df = df[df[self.feature].lt(self.end, fill_value=mean)]
        return df

    def jsonify(self):
        jsonified_dict = super(NumericalNode, self).jsonify()
        if self.beginning is None:
            jsonified_dict.pop("beginning")
        elif self.end is None:
            jsonified_dict.pop("end")
        return jsonified_dict

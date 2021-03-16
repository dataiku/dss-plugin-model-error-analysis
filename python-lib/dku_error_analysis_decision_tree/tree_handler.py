class TreeHandler(object):
    def __init__(self):
        self.set_tree(None)

    def set_tree(self, tree):
        self.selected_feature_ids = set()
        self.already_fetched_locally = set()
        self.already_fetched_globally = set()
        self.current_node_id = None
        self.tree = tree

    def set_current_node_id(self, node_id):
        self.current_node_id = node_id
        self.already_fetched_locally = set()

    def set_selected_feature_ids(self, feature_ids):
        new_ids = feature_ids - self.selected_feature_ids
        if self.current_node_id is not None:
            self.already_fetched_locally |= new_ids
        self.selected_feature_ids = feature_ids
        return new_ids - self.already_fetched_globally

    def get_stats_node(self, node_id):
        self.set_current_node_id(node_id)
        stats = {}
        for idx in self.selected_feature_ids:
            if idx not in self.already_fetched_locally:
                feature_name = self.tree.ranked_features[idx]["name"]
                stats[feature_name] = self.tree.get_stats(node_id, feature_name)
                self.already_fetched_locally.add(idx)
        return stats

    def get_stats_root(self, global_data_to_fetch):
        stats = {}
        for idx in global_data_to_fetch:
            feature_name = self.tree.ranked_features[idx]["name"]
            stats[feature_name] = self.tree.get_stats(0, feature_name)
            self.already_fetched_globally.add(idx)
        return stats
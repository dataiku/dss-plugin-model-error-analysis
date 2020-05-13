def _denormalize_feature_value(scalings, feature_name, feature_value):
    scaler = scalings.get(feature_name)
    if scaler is not None:
        inv_scale = scaler.inv_scale if scaler.inv_scale != 0.0 else 1.0
        return (feature_value / inv_scale) + scaler.shift
    else:
        return feature_value

def descale_numerical_thresholds(extract, feature_names, rescalers, is_regression):
    scalings = {rescaler.in_col: rescaler for rescaler in rescalers}
    features = extract.feature.tolist()
    def denormalize(feat, threshold):
        return threshold if feat < 0 else _denormalize_feature_value(scalings, feature_names[feat], threshold)

    return [denormalize(ft, thresh) for (ft, thresh) in zip(features, extract.threshold.tolist())]

    """tree = {
        "leftChild": extract.children_left.tolist(),
        "rightChild": extract.children_right.tolist(),
        "impurity": extract.impurity.tolist(),
        "threshold": thresholds,
        "nSamples": extract.weighted_n_node_samples.tolist(),
        "feature": features
    }
    if is_regression:
        tree["predict"] = [x[0][0] for x in  extract.value]
    else:
        tree["probas"] = [ [ u / y[1] for u in y[0]] for y in [(x[0], sum(x[0])) for x in extract.value]]
    return tree
    """

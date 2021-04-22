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

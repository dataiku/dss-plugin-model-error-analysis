# Model Error Analysis plugin

After training a ML model, data scientists need to investigate the model’s failures to build intuition around the critical subpopulations on which the model is performing poorly. This analysis is essential in the iterative process of model design and feature engineering, and is usually performed manually. 

The Model Error Analysis plugin provides the user with automatic tools to break down the model’s errors into meaningful groups, easier to analyze, and to highlight the most frequent type of errors, as well as the problematic characteristics correlated with the errors. 


## Scope of the plugin
This plugin offers a set of different DSS components to analyse the error of a deployed model:
* Model view: visualise the Error Tree and analyse the subgroups that conatain the most errors. 
* Template notebook: tutorial to use the python library and its api.

## Installation and requirements

Please see our [official plugin page](https://www.dataiku.com/product/plugins/model-error-analysis/) for installation.

## Changelog

**Version 1.0.0 (2021-04)**

* Initial release
* Model view component
* Template notebook component

You can log feature requests or issues on our [dedicated Github repository](https://github.com/dataiku/dss-plugin-model-error-analysis/issues).

# License

The Model Error Analysis plugin is:

   Copyright (c) 2020 Dataiku SAS
   Licensed under the [Apache](LICENSE.md).

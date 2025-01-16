# Changelog

## Version 1.3.3 (2025-01-16)
- Text and images embedding feature handling are now supported

## Version 1.3.2 (2024-05-14)
- Fix bug with TF-IDF column processing when used with sklearn > 1.0.0

## Version 1.3.1 (2023-11-01)
- Fix method used in the notebook template

## Version 1.3.0 (2023-05-04)
- Update code env description to support python versions 3.7, 3.8, 3.9, 3.10 and 3.11

## Version 1.2.0 (2023-04-19)
- Webapp: Use contextual code env (i.e. the one used to train the model)
- Webapp: Fix wrong handling of features generated from numerical quantile preprocessing (issue introduced in 1.1.3)

## Hotfix 1.1.4 (2022-07-25)
* Webapp: Fix typo

## Version 1.1.3 (2022-07-22)
* Webapp: Only display model view for tasks using a Python backend + of the proper prediction type
* Webapp: Use a better ModalService
* Webapp: Use a simpler way to load the models from DSS

## Version 1.1.2 (2022-01-13)

* Update code env requirements
* Fix issue with Lasso LARS on regression models

## Version 1.1.1 (2021-10)
* Fix name of tf idf preprocessed features not always recognized

## Version 1.1.0 (2021-10)
* Fix broken model view for models using quantization/binarization without Keep regular option

## Version 1.0.2 (2021-10)
* Add support for xgboost

## Version 1.0.0 (2021-04)

* Initial release
* Model view component
* Template notebook component

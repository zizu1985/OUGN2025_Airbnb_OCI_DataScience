Repository 
    This repository contains the code that was used to prepare the presentation "What shapes prices in Airbnb?" during the Oracle Norway User Group 2025 conference. .py files are Python files to run in a standard way. *.ipynb files are Jupyter notebooks for running in Colab, OCI DataScience or on-premises Jupyter notebooks installation. The presentation is in PowerPoint format.

Files
    automl/requirements.txt - required libraries to run examples (install with your Python Environment tool, I've used virtualenv)
    airbnb_ShallowTuner_BayesianOptimization_*.py - Select Shallow Model - Random Forest vs SVM, Use BayesianOptimization strategy
        berlin - automl/airbnb_ShallowTuner_BayesianOptimization_berlin.py
        munich - automl/airbnb_ShallowTuner_BayesianOptimization_munich.py
        prague - automl/airbnb_ShallowTuner_BayesianOptimization_prague.py
    airbnb_ShallowTuner_*.py -  Select Shallow Model - Random Forest vs SVM, Use RandomSearch strategy
        berlin - automl/airbnb_ShallowTuner_berlin.py
        munich - automl/airbnb_ShallowTuner_munich.py
        prague - automl/airbnb_ShallowTuner_prague.py
    MLP-Structured-Data-Regression.py - MLP and how we could treat exact architecture as hyperparameter
    airbnb_*_automl_taskapi.py- How to deal with AutoML Functional API
        Berlin - automl/airbnb_berlin_automl_taskapi.py
        Munich - automl/airbnb_munich_automl_taskapi.py
        Prague - automl/airbnb_prague_automl_taskapi.py
    Interpretable ML
        Berlin - automl/airbnb_explainable_AI_shap_berlin.ipynb
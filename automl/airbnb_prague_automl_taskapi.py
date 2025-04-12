'''
    This program demonstrates example of using AutoML Task Api.
    AutoML Task Api is simplest and quickest approach. Multiple ML models are evaluated based on
    constructed set by domain experts.
    Remember to clean up directory structured_data_regressor to erase previous training session.

    1. For all 96 columns MSE was
        Test MSE: 5438496256

        False             |structured_data_block_1/normalize
        False             |structured_data_block_1/dense_block_1/use_batchnorm
        2                 |structured_data_block_1/dense_block_1/num_layers
        32                |structured_data_block_1/dense_block_1/units_0
        0                 |structured_data_block_1/dense_block_1/dropout
        32                |structured_data_block_1/dense_block_1/units_1
        0                 |regression_head_1/dropout
        adam              |optimizer
        0.001             |learning_rate
        1024              |structured_data_block_1/dense_block_1/units_2

'''
##################################################################
# 0. Prereq steps
##################################################################
# Ignore all warning message from Tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import autokeras as ak
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ignore FutureWarnings messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import keras_tuner as kt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import matplotlib.pyplot as plt
plt.xkcd()

##################################################################
# 1. Loading input data
##################################################################
print('1. Loading input data')
prague_file='C:/Learning/Learning/Python/AutoKerasBook/data/listings_prague_11_2019.csv'

data_prague = pd.read_csv(prague_file)
print(f"size {data_prague.shape}")


##################################################################
# 2. Preprocess, enrich data and split for train/test
##################################################################
print('2. Preprocess, enrich data and split for train/test')

# Extract features with their names into a dataframe format
data_prague.price = data_prague.price.apply(lambda x: x.replace("$", ""))
data_prague.price = data_prague.price.apply(lambda x: x.replace(",", ""))
data_prague.price = data_prague.price.astype("float")
data_prague['number_of_reviews'] = data_prague['number_of_reviews'].astype(float)
data_prague['accommodates'] = data_prague['accommodates'].astype(float)
data_prague['minimum_nights'] = data_prague['minimum_nights'].astype(float)

data_subset = data_prague

data = pd.DataFrame(data_subset)

# Extract target with their names into a pd.Series object with name MEDV
target = pd.Series(data_subset['price'], name="Price")

train_data, test_data, train_targets, test_targets = train_test_split(
    data, target, test_size=0.2
)

##################################################################
# 3. Find best model and hyperparameters
##################################################################
import autokeras as ak
regressor = ak.StructuredDataRegressor(max_trials=30)
regressor.fit(x=train_data, y=train_targets,
    batch_size=16, verbose=2)
test_loss, test_mse = regressor.evaluate(
    test_data, test_targets, verbose=2)
print('Test MSE: ', test_mse)

'''
    This program demonstrates example of using AutoML Task Api.
    AutoML Task Api is simplest and quickest approach. Multiple ML models are evaluated based on
    constructed set by domain experts.
    Remember to clean up directory structured_data_regressor to erase previous training session.

    1. For selected 12 columns MSE was
        Test MSE:  7044

        Value             |Best Value So Far |Hyperparameter
        True              |True              |structured_data_block_1/normalize
        False             |False             |structured_data_block_1/dense_block_1/use_batchnorm
        3                 |3                 |structured_data_block_1/dense_block_1/num_layers
        512               |512               |structured_data_block_1/dense_block_1/units_0
        0                 |0                 |structured_data_block_1/dense_block_1/dropout
        16                |32                |structured_data_block_1/dense_block_1/units_1
        0.25              |0.25              |regression_head_1/dropout
        adam              |adam              |optimizer
        0.001             |0.001             |learning_rate
        16                |16                |structured_data_block_1/dense_block_1/units_2

    2. For all 96 columns MSE was
        Test MSE:  5526

        Value             |Best Value So Far |Hyperparameter
        True              |True              |structured_data_block_1/normalize
        False             |False             |structured_data_block_1/dense_block_1/use_batchnorm
        2                 |2                 |structured_data_block_1/dense_block_1/num_layers
        32                |32                |structured_data_block_1/dense_block_1/units_0
        0.25              |0.25              |structured_data_block_1/dense_block_1/dropout
        16                |16                |structured_data_block_1/dense_block_1/units_1
        0.5               |0.5               |regression_head_1/dropout
        adam              |adam              |optimizer
        0.001             |0.001             |learning_rate
        1024              |None              |structured_data_block_1/dense_block_1/units_2

'''
##################################################################
# 0. Prerequisities steps
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
munich_file='C:/Learning/Learning/Python/AutoKerasBook/data/listings_munich_11_2019.csv'

data_munich = pd.read_csv(munich_file)
print(f"size {data_munich.shape}")

##################################################################
# 2. Preprocess, enrich data and split for train/test
##################################################################
print('2. Preprocess, enrich data and split for train/test')

# Extract features with their names into a dataframe format
data_munich.price = data_munich.price.apply(lambda x: x.replace("$", ""))
data_munich.price = data_munich.price.apply(lambda x: x.replace(",", ""))
data_munich.price = data_munich.price.astype("float")
data_munich['number_of_reviews'] = data_munich['number_of_reviews'].astype(float)
data_munich['accommodates'] = data_munich['accommodates'].astype(float)
data_munich['minimum_nights'] = data_munich['minimum_nights'].astype(float)

# Enrich data by calculate zip code relation to price
# amenities_len -> value for comparing number of amenities across offerts
# zip_count -> number of offerts in close location
# zip_price -> average price for offerts in the same location

temp_zipcode = data_munich.zipcode.copy()
data_munich['zipcode2'] = temp_zipcode.str.replace("\D+", "", ).copy()

data_munich.zipcode2.fillna(0, inplace=True)

x_count = data_munich.groupby('zipcode2')['id'].nunique()
x_mean = data_munich.groupby('zipcode2')['price'].mean()

x_count_dict = x_count.to_dict()
x_mean_dict = x_mean.to_dict()

a1 = np.zeros((len(data_munich), 6))
print(a1)
for i in range(0,len(data_munich)):
    val = data_munich.zipcode2[i]
    a1[i][0] = data_munich.id[i]
    a1[i][1] = x_count_dict[val]
    a1[i][2] = x_mean_dict[val]
    a1[i][3] = val
    a1[i][4] = len(data_munich.amenities[i])

data_munich['amenities_len'] = a1[:,3]
data_munich['zipcode_count'] = a1[:,1]
data_munich['zipcode_price'] = a1[:,2]
print(data_munich.head())

# Preprocessing - get rid of outliers
print("99.7% properties have a price lower than {0: .2f}".format(np.percentile(data_munich.price, 99.7)))
data_munich = data_munich[(data_munich.price <= np.percentile(data_munich.price, 99.7)) & (data_munich.price > 0)]

cols = ['price', 'host_is_superhost', 'bedrooms', 'number_of_reviews', 'review_scores_rating', 'beds', 'bathrooms']
cols2 = ['accommodates', 'amenities_len', 'minimum_nights', 'zipcode_count', 'zipcode_price']
cols = cols + cols2

data_munich.fillna((data_munich[cols].mean()), inplace=True)
data_munich['number_of_reviews'] = data_munich['number_of_reviews'].astype(float)
data_munich['accommodates'] = data_munich['accommodates'].astype(float)
data_munich['amenities_len'] = data_munich['amenities_len'].astype(float)
data_munich['minimum_nights'] = data_munich['minimum_nights'].astype(float)


"""
variables = ['amenities_len','accommodates','bedrooms','beds','bathrooms','zipcode_count','zipcode_price','number_of_reviews','review_scores_rating', 'guests_included', 'minimum_nights']

data_subset = data_munich.loc[:, variables + ['price']]
data_subset.dropna(inplace=True)

data_munich_data= data_subset.loc[:,variables + ['price']]
data = pd.DataFrame(data_munich_data)

# Extract target with their names into a pd.Series object with name MEDV
target = pd.Series(data_subset['price'], name="Price")

train_data, test_data, train_targets, test_targets = train_test_split(
    data, target, test_size=0.2
)
"""

data_munich = data_munich[:8000]
data_subset = data_munich

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
regressor = ak.StructuredDataRegressor(max_trials=20)
regressor.fit(x=train_data, y=train_targets,
    batch_size=16, verbose=2)
test_loss, test_mse = regressor.evaluate(
    test_data, test_targets, verbose=2)
print('Test MSE: ', test_mse)

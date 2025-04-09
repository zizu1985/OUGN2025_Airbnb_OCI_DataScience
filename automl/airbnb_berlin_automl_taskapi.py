'''
    This program demonstrates example of using AutoML Task Api.
    AutoML Task Api is simplest and quickest approach. Multiple ML models are evaluated based on
    constructed set by domain experts.
    Remember to clean up directory structured_data_regressor to erase previous training session.

    1. For selected 12 columns MSE was
        Test MSE:  929

        with final model architecture

        ######### Architecture #########################
        Best Value So Far |Hyperparameter
        True              |structured_data_block_1/normalize
        False             |structured_data_block_1/dense_block_1/use_batchnorm
        2                 |structured_data_block_1/dense_block_1/num_layers
        512               |structured_data_block_1/dense_block_1/units_0
        0.5               |structured_data_block_1/dense_block_1/dropout
        32                |structured_data_block_1/dense_block_1/units_1
        0.5               |regression_head_1/dropout
        adam              |optimizer
        0.001             |learning_rate
        32                |structured_data_block_1/dense_block_1/units_2
        ###################################################

    2. For all 96 columns MSE was
        Test MSE:  209

         with final model architecture
        ######### Architecture #########################
        Best Value So Far |Hyperparameter
        True              |structured_data_block_1/normalize
        True              |structured_data_block_1/dense_block_1/use_batchnorm
        2                 |structured_data_block_1/dense_block_1/num_layers
        512               |structured_data_block_1/dense_block_1/units_0
        0                 |structured_data_block_1/dense_block_1/dropout
        32                |structured_data_block_1/dense_block_1/units_1
        0.5               |regression_head_1/dropout
        adam              |optimizer
        0.01              |learning_rate
        64                |structured_data_block_1/dense_block_1/units_2
        ###################################################


'''
# Ignore all warning message comming from Tensorflow
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
berlin_file='C:/Learning/Learning/Python/AutoKerasBook/data/listings_berlin_11_2019.csv'

data_berlin = pd.read_csv(berlin_file)
print(f"size {data_berlin.shape}")


##################################################################
# 2. Preprocess, enrich data and split for train/test
##################################################################
print('2. Preprocess, enrich data and split for train/test')

# Extract features with their names into a dataframe format
data_berlin.price = data_berlin.price.apply(lambda x: x.replace("$", ""))
data_berlin.price = data_berlin.price.apply(lambda x: x.replace(",", ""))
data_berlin.price = data_berlin.price.astype("float")
data_berlin['number_of_reviews'] = data_berlin['number_of_reviews'].astype(float)
data_berlin['accommodates'] = data_berlin['accommodates'].astype(float)
data_berlin['minimum_nights'] = data_berlin['minimum_nights'].astype(float)

# on-premises, resources limitation
data_berlin = data_berlin[:1000]

print('First 5 rows in data')
pd.set_option('display.max_columns', 3)
print(data_berlin.head())

# Enrich data by adding amenities_len, zipcode_count and zipcode_price
temp_zipcode = data_berlin.zipcode.copy()
data_berlin['zipcode2'] = temp_zipcode.str.replace("\D+", "", ).copy()
data_berlin.zipcode2.fillna(0, inplace=True)
x_count = data_berlin.groupby('zipcode2')['id'].nunique()
x_mean = data_berlin.groupby('zipcode2')['price'].mean()

x_count_dict = x_count.to_dict()
x_mean_dict = x_mean.to_dict()

a1 = np.zeros((len(data_berlin), 6))
for i in range(0,len(data_berlin)):
    val = data_berlin.zipcode2[i]
    a1[i][0] = data_berlin.id[i]
    a1[i][1] = x_count_dict[val]
    a1[i][2] = x_mean_dict[val]
    a1[i][3] = val
    a1[i][4] = len(data_berlin.amenities[i])

data_berlin['amenities_len'] = a1[:,3]
data_berlin['zipcode_count'] = a1[:,1]
data_berlin['zipcode_price'] = a1[:,2]


'''
variables = ['amenities_len','accommodates','bedrooms','beds','bathrooms','zipcode_count','zipcode_price','number_of_reviews','review_scores_rating', 'guests_included', 'minimum_nights']

data_subset = data_berlin.loc[:, variables + ['price']]
data_subset.dropna(inplace=True)

data_berlin_data= data_subset.loc[:,variables + ['price']]
data = pd.DataFrame(data_berlin_data)

# Extract target with their names into a pd.Series object with name MEDV
target = pd.Series(data_subset['price'], name="Price")

train_data, test_data, train_targets, test_targets = train_test_split(
    data, target, test_size=0.2
)
'''

data_subset = data_berlin

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

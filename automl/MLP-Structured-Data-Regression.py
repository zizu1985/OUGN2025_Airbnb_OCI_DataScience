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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import keras_tuner as kt
import tensorflow as tf
import os
import pickle

# Required for generating model plots
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

##################################################################
# 1. Loading input data
##################################################################
print('1. Loading input data')
berlin_file='C:/Learning/Learning/Python/AutoKerasBook/data/listings_berlin_11_2019.csv'

data_berlin = pd.read_csv(berlin_file)
print(f"size {data_berlin.shape}")

##################################################################
# 2. Create AutoML pipeline for StructureData Regression analysis
##################################################################
print('2. Building AutoML pipeline')
input_node = ak.StructuredDataInput()
output_node = ak.Normalization()(input_node)
output_node = ak.DenseBlock(use_batchnorm=False)(output_node)
output_node = ak.RegressionHead()(output_node)
auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, max_trials=10, overwrite=True, seed=51
)

##################################################################
# 3. Preprocess, enrich data and split for train/test
##################################################################
print('3. Preprocess, enrich data and split for train/test')

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

print(test_data)
print(test_targets)
##################################################################
# 4. Train data
##################################################################
print('4. Train model')
# Batch size reduced to allow training
auto_model.fit(train_data, train_targets, batch_size=32, epochs=150)

# Data reduced to first 1000 rows
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 32.5 GiB for an array with shape (18041, 95) and data type <U5095

##################################################################
# 5. Evaluate best pipeline and show the best trial
##################################################################
print('5. Test accuracy')
test_loss, test_acc = auto_model.evaluate(test_data, test_targets, verbose=1)
print('Accuracy: {accuracy}%'.format(accuracy=round(test_acc*100,2)))

best_model = auto_model.export_model()

# Show best trial
from tensorflow import keras

best_model.save("saved_model")
best_model = keras.models.load_model("saved_model")

# Display best model
best_model = auto_model.export_model()
tf.keras.utils.plot_model(
    best_model, show_shapes=True, expand_nested=True
)  # rankdir='LR'

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

traint_data = train_data[:1000]
test_data = test_data[:100]
train_targets = train_targets[:1000]
test_targets = test_targets[:100]

print(test_data)
print(test_targets)


##################################################################
# 3. Create ShallowTuner
##################################################################
class ShallowTuner(kt.Tuner):
    def __init__(self, oracle, hypermodel, **kwargs):
        super(ShallowTuner, self).__init__(
            oracle=oracle, hypermodel=hypermodel, **kwargs
        )

    def search(self, X, y, validation_data):
        """performs hyperparameter search."""
        return super(ShallowTuner, self).search(X, y, validation_data)

    def run_trial(self, trial, X, y, validation_data):
        model = self.hypermodel.build(trial.hyperparameters)  # build the model
        model.fit(X, y)  # fit the model
        X_val, y_val = validation_data  # get the validation data
        eval_score = model.score(X_val, y_val)  # evaluate the model
        self.save_model(trial.trial_id, model)  # save the model to disk
        return {"score": eval_score}

    def save_model(self, trial_id, model, step=0):
        """save the model with pickle"""
        fname = os.path.join(self.get_trial_dir(trial_id), "model.pickle")
        with tf.io.gfile.GFile(fname, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, trial):
        """load the model with pickle"""
        fname = os.path.join(self.get_trial_dir(trial.trial_id), "model.pickle")
        with tf.io.gfile.GFile(fname, "rb") as f:
            return pickle.load(f)

def build_model(hp):
    model_type = hp.Choice("model_type", ["svm", "random_forest"])
    if model_type == "svm":
        with hp.conditional_scope("model_type", "svm"):
            model = SVC(
                C=hp.Float("C", 1e-3, 10, sampling="linear", default=1),
                kernel=hp.Choice("kernel_type", ["linear", "rbf"], default="linear"),
                random_state=42,
            )
    elif model_type == "random_forest":
        with hp.conditional_scope("model_type", "random_forest"):
            model = RandomForestClassifier(
                n_estimators=hp.Int("n_estimators", 10, 200, step=10),
                max_depth=hp.Int("max_depth", 3, 10),
            )
    else:
        raise ValueError("Unrecognized model_type")
    return model

my_sklearn_tuner = ShallowTuner(
    oracle=kt.oracles.RandomSearch(
        objective=kt.Objective("score", "max"), max_trials=10, seed=42
    ),
    hypermodel=build_model,
    overwrite=True,
    project_name="my_sklearn_tuner",
)

##################################################################
# 3. Search for best hyperparameter set
##################################################################
my_sklearn_tuner.search(train_data, train_targets, validation_data=(test_data, test_targets))
print('Best hyperparameter')
print((my_sklearn_tuner.get_best_hyperparameters()[0]).values)


##################################################################
# 4. Evaluate best model
##################################################################
from sklearn.metrics import accuracy_score

best_model = my_sklearn_tuner.get_best_models(1)[0]
best_model.fit(traint_data, train_targets)
y_pred_test = best_model.predict(test_data)
test_acc = accuracy_score(test_targets, y_pred_test)
print("The prediction accuracy on test set: {:.2f} %".format(test_acc * 100))
# -*- coding: utf-8 -*-
"""HDP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bkhYmD-kvZ3GvkNRDW7AwAfub_t0D1Uv
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

"""Reading the Data"""

!wget https://raw.githubusercontent.com/Ramesh307/data/main/data.csv

"""Data Description
age--age in years 2 .sex-- (1)=Male (0)=Female
cp ==> chest_pain_type
trestbps==>resting blood pressure (in mm Hg on admission to the hospital)
chol==>serum cholestoral in mg/dl
fbs==>( fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg==>resting electrocardiographic results
thalach==>maximum heart rate achieved
exang==>exercise induced angina (1 = yes; 0 = no)
oldpeak==>ST depression induced by exercise relative to rest
slope==>The slope of the Peak exercise ST segment
ca==> number of major vessels(0-3) colored by flourosopy
thal==>1=> normal,2=>fixed defect,3=>reversable defect
"""

df = pd.read_csv("data.csv", na_values = "?")
df_copy = df.copy()

df_copy=df_copy.rename(columns={"num       ":"target"})

df_copy=df_copy.drop(columns=["slope","ca","thal"],axis=1)

df_copy.head(10)

df_copy["cp"] = df_copy["cp"].astype("float32")

df_copy["restecg"] = df_copy["restecg"].astype("float32")

df_copy["fbs"] = df_copy["fbs"].astype("float32")

df_copy["exang"] = df_copy["exang"].astype("float32")

df_copy["restecg"].fillna(0.0, inplace = True)

df_copy["exang"].fillna(0, inplace = True)

df_copy["fbs"].fillna(0, inplace = True)

df_copy["cp"] = df_copy["cp"].astype("float32")

df_copy.shape

median_trestbps = df_copy["trestbps"].median()
median_chol = df_copy["chol"].median()
median_thalach = df_copy["thalach"].median()

df_copy["trestbps"].fillna(median_trestbps, inplace = True)
df_copy["chol"].fillna(median_chol, inplace = True)
df_copy["thalach"].fillna(median_thalach, inplace = True)

df_copy.isnull().sum()

df_copy.head()

X = df_copy.drop("target", axis = 1)
y = df_copy["target"]

df_copy["exang"] = df_copy["exang"].astype("float32")

df_copy["fbs"]=df_copy["fbs"].astype("float32")

df_copy.info()

plt.figure(figsize=(15,10))
ax = sns.boxplot(data=df_copy)

df_copy=df_copy.drop_duplicates()

from scipy import stats
z = np.abs(stats.zscore(df_copy))
print(z)

threshold = 3
print(np.where(z > 3))

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR=Q3-Q1
print(IQR)

df_copy = df_copy[(z < 3).all(axis=1)]
df.shape

df_copy = df_copy[(z < 3).all(axis=1)]
df.shape

df_copy.isnull().sum() / len(df)

df_copy = df_copy[~((df_copy < (Q1 - 1.5 * IQR)) |(df > Q3 + 1.5 * IQR)).any(axis=1)]
df_copy.shape

plt.figure(figsize=(25,20))
ax = sns.boxplot(data=df_copy)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()

y_train.head()

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)

# X_test = sc.fit_transform(X_test)

# from sklearn.preprocessing import StandardScaler
# StandardScaler = StandardScaler()
# dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
# columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# dataset[columns_to_scale] = StandardScaler.fit_transform(dataset[columns_to_scale])

#@title Default title text
df_copy.describe

"""# This is formatted as code

Algorithm Implementation
"""

def entropy(p):
    if p == 0:
        return 0
    elif p == 1:
        return 0
    else:
        return - (p * np.log2(p) + (1 - p) * np.log2(1-p))

def information_gain(left_child, right_child):
    parent = left_child + right_child
    p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
    p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
    p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
    IG_p = entropy(p_parent)
    IG_l = entropy(p_left)
    IG_r = entropy(p_right)
    return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r

# bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
# X_bootstrap = X_train[bootstrap_indices,:]

# bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))

bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
X_bootstrap = X_train.iloc[bootstrap_indices].values

y_train.iloc[bootstrap_indices]

import numpy as np

np.random.seed(10)

def draw_bootstrap(X_train, y_train):
    np.random.seed(10)
    bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
    oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
    X_bootstrap = X_train.iloc[bootstrap_indices].values
    y_bootstrap = y_train.iloc[bootstrap_indices].values
    X_oob = X_train.iloc[oob_indices].values
    y_oob = y_train.iloc[oob_indices].values
    return X_bootstrap, y_bootstrap, X_oob, y_oob

def oob_score(tree, X_test, y_test):
    mis_label = 0
    for i in range(len(X_test)):
        pred = predict_tree(tree, X_test[i])
        if pred != y_test[i]:
            mis_label += 1
    return mis_label / len(X_test)

import random
def find_split_point(X_bootstrap, y_bootstrap, max_features):
    np.random.seed(10)
    feature_ls = list()
    num_features = len(X_bootstrap[0])

    while len(feature_ls) <= max_features:
      feature_idx = random.sample(range(num_features), 1)
      if feature_idx not in feature_ls:
        feature_ls.extend(feature_idx)

    best_info_gain = -999
    node = None
    for feature_idx in feature_ls:
      for split_point in X_bootstrap[:,feature_idx]:
        left_child = {'X_bootstrap': [], 'y_bootstrap': []}
        right_child = {'X_bootstrap': [], 'y_bootstrap': []}

        # split children for continuous variables
        if type(split_point) in [int, float]:
            for i, value in enumerate(X_bootstrap[:,feature_idx]):
                if value <= split_point:
                    left_child['X_bootstrap'].append(X_bootstrap[i])
                    left_child['y_bootstrap'].append(y_bootstrap[i])
                else:
                    right_child['X_bootstrap'].append(X_bootstrap[i])
                    right_child['y_bootstrap'].append(y_bootstrap[i])
        # split children for categoric variables
        else:
            for i, value in enumerate(X_bootstrap[:,feature_idx]):
                if value == split_point:
                    left_child['X_bootstrap'].append(X_bootstrap[i])
                    left_child['y_bootstrap'].append(y_bootstrap[i])
                else:
                    right_child['X_bootstrap'].append(X_bootstrap[i])
                    right_child['y_bootstrap'].append(y_bootstrap[i])

        split_info_gain = information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
        if split_info_gain > best_info_gain:
            best_info_gain = split_info_gain
            left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
            right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
            node = {'information_gain': split_info_gain,
                    'left_child': left_child,
                    'right_child': right_child,
                    'split_point': split_point,
                    'feature_idx': feature_idx}


    return node

def terminal_node(node):
    np.random.seed(10)
    y_bootstrap = node['y_bootstrap']
    pred = max(y_bootstrap, key = y_bootstrap.count)
    return pred


def split_node(node, max_features, min_samples_split, max_depth, depth):
    np.random.seed(10)
    left_child = node['left_child']
    right_child = node['right_child']

    del(node['left_child'])
    del(node['right_child'])

    if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
        empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
        node['left_split'] = terminal_node(empty_child)
        node['right_split'] = terminal_node(empty_child)
        return

    if depth >= max_depth:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return node

    if len(left_child['X_bootstrap']) <= min_samples_split:
        node['left_split'] = node['right_split'] = terminal_node(left_child)
    else:
        node['left_split'] = find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
        split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
    if len(right_child['X_bootstrap']) <= min_samples_split:
        node['right_split'] = node['left_split'] = terminal_node(right_child)
    else:
        node['right_split'] = find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
        split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

def build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
    np.random.seed(10)
    root_node = find_split_point(X_bootstrap, y_bootstrap, max_features)
    split_node(root_node, max_features, min_samples_split, max_depth, 1)
    return root_node

def random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    np.random.seed(10)
    tree_ls = list()
    oob_ls = list()
    for i in range(n_estimators):
        
        X_bootstrap, y_bootstrap, X_oob, y_oob = draw_bootstrap(X_train, y_train)
       
        tree = build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
        tree_ls.append(tree)
        oob_error = oob_score(tree, X_oob, y_oob)
        oob_ls.append(oob_error)
    print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls

def predict_tree(tree, X_test):
    np.random.seed(10)
    feature_idx = tree['feature_idx']

    if X_test[feature_idx] <= tree['split_point']:
        if type(tree['left_split']) == dict:
            return predict_tree(tree['left_split'], X_test)
        else:
            value = tree['left_split']
            return value
    else:
        if type(tree['right_split']) == dict:
            return predict_tree(tree['right_split'], X_test)
        else:
            return tree['right_split']

def predict_rf(tree_ls, X_test):
    np.random.seed(10)
    pred_ls = list()
    for i in range(len(X_test)):
        ensemble_preds = [predict_tree(tree, X_test.iloc[i]) for tree in tree_ls]
        final_pred = max(ensemble_preds, key = ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)

n_estimators = 100
max_features = 3
max_depth = 10
min_samples_split = 2

import time
start_time = time.time()
model = random_forest(X_train, y_train, n_estimators=200, max_features=1, max_depth=9, min_samples_split=2)
end_time = time.time()
total_time = end_time - start_time
print(f"Training Time :{total_time}")
## Training prediction
preds = predict_rf(model, X_train)
acc = sum(preds == y_train) / len(y_train)
print("Training accuracy : {}".format(np.round(acc, 3)))


##Testing Prediction
preds = predict_rf(model, X_test)
acc = sum(preds == y_test) / len(y_test)
print("Testing accuracy: {}".format(np.round(acc,3)))

###

"""Saving the trained model"""

import pickle
import numpy as np

#filename= 'trained_model.sav'
#pickle.dump(predict_rf, open(filename, 'wb'))

"""Loading the saved Model"""

#loaded_model= pickle.load(open("trained_model.sav", "rb"))

# Define a function to make predictions using the loaded model
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak):
    # Convert the input values to a numpy array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])

    # Use the loaded model to make predictions on the input data
    prediction = predict_rf(input_data)

    # Return the predicted class (0 or 1)
    print(prediction)

    if(prediction[0]==0):
      print("You have Heart Disease")
    else:
      print("You dont have")



def predict_str(tree_ls, X_test):
    np.random.seed(10)
    pred_ls = list()
    for i in range(len(X_test)):
        ensemble_preds = [predict_tree(tree, X_test[i]) for tree in tree_ls]
        final_pred = max(ensemble_preds, key = ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)

#input_data=(41,0,1,130,204,0,0,172,0,1.4,2,0,2)

#changing the input data to numpy array
#input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance
#input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
#prediction=pickle.load(loaded_model)
#print(prediction)

#if (loaded_model[0] == 0):
 # print('You have Heart Disease')
#else:
 # print('You don\'t have Heart Disease')

# Define a function to make predictions using the loaded model
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak):
    # Convert the input values to a numpy array
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])

    # Use the loaded model to make predictions on the input data
    prediction = predict_str(input_data)

    # Return the predicted class (0 or 1)
    print(prediction)

    if(prediction[0]==0):
      print("You have Heart Disease")
    else:
      print("You dont have")

#predict_heart_disease(63,1,3,145,233,1,0,150,0,2.3)


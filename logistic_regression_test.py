import os
import random
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow
from skimage.io import imread
from data_load import load_train
from data_load import load_image
from data_load import load_labels
from data_load import load_image_min
import data_load
import seaborn as sns


df = load_train()
label_list = load_labels()

Y_train = df['tags'].values
X_train = []
#input near infrared
for index, row in df.iterrows():
    img = load_image_min(row['image_name'] + '.tif')
    nir = img[:, :, 3]
    X_train.append(nir.flatten())

X_train = np.array(X_train)
print(X_train.shape)

from sklearn.decomposition import SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_dev, X_val, y_dev, y_val = train_test_split(X_train, Y_train, train_size=0.8, random_state=0)
print(y_dev)
print(y_dev.shape)

pca = SparsePCA(n_components=100, n_jobs=-1)
pca.fit(X_dev[:100])
X_dev_pca = pca.transform(X_dev)
X_val_pca = pca.transform(X_val)
joblib.dump(pca, 'models/pca.pkl')

temp = np.array([line for line in y_dev])
print(temp[:, 1])
print(temp.shape)
print(temp[:, 1].shape)

y_dev = np.array([line for line in y_dev])
y_val = np.array([line for line in y_val])

print(y_dev.shape)
print(y_val.shape)

from sklearn.externals import joblib

for i in range(17):

    lr = LogisticRegression()
    lr.fit(X_dev_pca, y_dev[:, i])
    y_pred = lr.predict_proba(X_val_pca)
    score = roc_auc_score(y_val[:,i], y_pred[:, 1])
    print(i)
    print(score)
    joblib.dump(lr, 'models/lr-' + str(i) +'.pkl')

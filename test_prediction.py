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
from sklearn.externals import joblib
import glob
from sklearn.decomposition import SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from skimage import io
from skimage import transform
from data_load import load_test_image

filenames = glob.glob('input/test-jpg/*') + glob.glob('input/test-jpg-additional/*')
print(len(filenames))
filenames = np.array(filenames)
image_count = len(filenames)

for i in range(image_count):
    filenames[i] = filenames[i].split('\\')[1].split('.')[0]

X_test = []
#input near infrared
for filename in filenames:
    img = load_test_image(filename + '.jpg')
    nir = img[:, :, 2]
    X_test.append(nir.flatten())

print(X_test)
X_test = np.array(X_test)
print(X_test.shape)

pca = joblib.load('models/pca.pkl')
X_test_pca = pca.transform(X_test)
y_pred = []
y_pred.append(filenames)
for i in range(17):

    lr = joblib.load('models/lr-' + str(i) +'.pkl')
    y_pred_ = lr.predict_proba(X_test_pca)
    y_pred_ = y_pred_[:,1]
    y_pred.append(y_pred_)

print(y_pred)
y_pred = np.array(y_pred)
print(y_pred)
y_pred = y_pred.T
print(y_pred)

from data_load import CLASS_INDEX_TO_LABEL

y_submit = []
for row in y_pred:
    #print(row[12])
    #break
    string = ''
    for i in range(1,18):
        if float(row[i]) > 0.5:
            string += CLASS_INDEX_TO_LABEL[i - 1] + ' '
    y_submit.append([row[0], string])

print(y_submit)

df = pd.DataFrame(y_submit, columns=['image_name','tags'])
print(df)

'''
transform tags_vec to string
'''
df.to_csv("submission.csv", index=False)

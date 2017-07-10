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
import data_load
import seaborn as sns

df = load_train()
label_list = load_labels()
sum_tags = df.tags.sum()

plt.plot(sum_tags[np.argsort(sum_tags)])
plt.xticks(np.argsort(sum_tags))
plt.show()

sorted_index = np.argsort(sum_tags)
numeric_df = np.array([line[sorted_index] for line in df.tags])
c_matrix = numeric_df.T.dot(numeric_df)

ax = sns.heatmap(c_matrix)
sorted_labels = [data_load.CLASS_INDEX_TO_LABEL[i] for i in np.argsort(sum_tags)]
ax.set_xticklabels(sorted_labels, rotation=90, fontsize="small")
plt.show()


'''
No correration between any weather labels
land labels overlaped
rarer label have very little overlapping
'''

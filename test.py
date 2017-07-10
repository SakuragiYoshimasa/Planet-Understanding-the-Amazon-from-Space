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

def main():

    print(df)

    for index, row in df.iterrows():
        img = load_image(row['image_name'] + '.tif')
        print(type(img)) #np.ndarray
        print(img.shape)
        print(row['image_name'])
        r, g, b, nir = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        fig = plt.figure()
        fig.set_size_inches(12, 4)
        for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'))):
            a = fig.add_subplot(1, 4, i+1)
            a.set_title(c)
            plt.imshow(x)
            plt.gray()
        plt.show()
        if index > 1:break
    Y_train = df['tags'].values

if __name__ == '__main__':
    main()

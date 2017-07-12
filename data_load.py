import os
import pandas as pd
import numpy as np
from skimage import io
import rasterio
from skimage import io
from skimage import transform

CLASS_LABEL_TO_INDEX = {
    'agriculture':0, 'artisinal_mine':1, 'bare_ground':2, 'blooming':3, 'blow_down':4,
    'clear':5, 'cloudy':6, 'conventional_mine':7, 'cultivation':8, 'habitation':9,
    'haze':10, 'partly_cloudy':11, 'primary':12, 'road':13, 'selective_logging':14,
    'slash_burn':15, 'water':16
    }
CLASS_INDEX_TO_LABEL = {CLASS_LABEL_TO_INDEX[key]: key for key in CLASS_LABEL_TO_INDEX.keys()}
TAG_COUNT = 17

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Load train_v2.csv and tags are transformed into vector
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def load_train():
    df = pd.read_csv('input/train_v2.csv')
    #image_name,tags
    tag_count = 17
    for index, row in df.iterrows():
        tags_vec = np.zeros(tag_count, dtype=np.float32)
        tags = row['tags']
        tags = tags.split(' ')
        for tag in tags: tags_vec[CLASS_LABEL_TO_INDEX[tag]] = 1.0
        row['tags'] = tags_vec
    return df


def load_image(img_name):

    if 'tif' in img_name:
        path = os.path.join('input/train-tif-v2', img_name)
        return io.imread(path)
    else:
        path = os.path.join('input/train-jpg', img_name)
        img = io.imread(path)
        return img

def load_image_min(img_name):
    if 'tif' in img_name:
        path = os.path.join('input/train-tif-v2', img_name)
        img = io.imread(path)
        img = transform.resize(img, (100, 100), mode='reflect')
        return img
    else:
        path = os.path.join('input/train-jpg', img_name)
        img = io.imread(path)
        img = transform.resize(img, (100, 100), mode='reflect')
        return img

def load_test_image(img_name):
    if 'tif' in img_name:
        path = os.path.join('input/fixed', img_name)
        img = io.imread(path)
        img = transform.resize(img, (100, 100), mode='reflect')
        return img
    else:
        if 'file' in img_name:
            path = os.path.join('input/test-jpg-additional', img_name)
        else:
            path = os.path.join('input/test-jpg', img_name)

        img = io.imread(path)
        img = transform.resize(img, (100, 100), mode='reflect')
        return img

def load_labels():
    return [label for label in CLASS_LABEL_TO_INDEX.keys()]


if __name__ == '__main__':
    print(load_train())

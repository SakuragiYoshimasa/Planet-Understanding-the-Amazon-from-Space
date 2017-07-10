import os
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

BASEPATH = 'input'


CSVPATH = os.path.join(BASEPATH, 'test_v2_file_mapping.csv')
JPGPATH = os.path.join(BASEPATH, 'test-jpg-v2')
TIFPATH = os.path.join(BASEPATH, 'test-tif-v2')

FIXEDPATH = os.path.join(BASEPATH, 'fixed')

def copy_and_rename(num_files=500):
    '''Copy up to `num_files` images to the scratch directory.
    `num_files` is needed because you can only write a few hundred
    megabytes in this kernel environment. Use the `df -h` command
    to check.

    This is a purposely non-destructive operation. You'll need to
    move the renamed files back to the test-tif-v2 directory so
    that your existing scripts will continue to work.
    '''
    #n = 0
    df = pd.read_csv(CSVPATH)
    if not os.path.exists(FIXEDPATH):
        os.mkdir(FIXEDPATH)

    for index, row in df.iterrows():
        old = os.path.join(TIFPATH, row['old'])
        new = os.path.join(FIXEDPATH, row['new'])
        shutil.copy(old, new)

        #n += 1
        if index % 500 == 0:
            print('Copied {}'.format(index))


if __name__ == '__main__':
    copy_and_rename()

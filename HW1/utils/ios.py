import os
import glob

from typing import List

import pandas as pd
import numpy as np


#%% functions
def read_single_file(path: str) -> pd.DataFrame:
    """
    Reads only a file containing the review

    :param path: path to the file
    :return: a pandas `DataFrame`
    """
    with open(path) as file:
        review = [file.read()]
    return pd.DataFrame(review, columns=['text'])


def read_all_files(path: str, shuffle: bool = True) -> pd.DataFrame:
    """
    Reads all files in the the given folder `path`

    :param path: A str "path/" to the folder containing all files ("data/train") sub folder of classes will be handled
    :param shuffle: shuffle the data order or not
    :return: a pandas `DataFrame['text', 'label']`
    """

    pos_reviews = []
    neg_reviews = []

    root_path = path
    pos_path = root_path + '/pos/*.txt'
    files = glob.glob(pos_path)

    if len(files) == 0:
        raise Exception('The path:"{}" contains no file to be read.'.format(path))

    for name in files:
        with open(name) as f:
            pos_reviews.append(f.read())
    neg_path = root_path + '/neg/*.txt'
    files = glob.glob(neg_path)
    for name in files:
        with open(name) as f:
            neg_reviews.append(f.read())

    labels = np.zeros(len(pos_reviews + neg_reviews), dtype=np.int)
    labels[0:len(pos_reviews)] = 1

    pos_reviews.extend(neg_reviews)
    del neg_reviews

    data_frame = pd.DataFrame(pos_reviews, columns=['text'])
    data_frame['label'] = labels

    if shuffle:
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)  # note the operation is in-place
        return data_frame
    return data_frame


#%% tests
if __name__ == '__main__':
    root = 'sentiment_analysis/data/sample/train/'
    file_path = root+'pos/0_9.txt'
    df_single = read_single_file(file_path)
    df = read_all_files(root, True)

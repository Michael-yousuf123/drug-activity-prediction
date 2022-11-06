import re
import os
from pathlib import Path
import numpy as np 
import pandas as pd
import configparser
#------------------------------------------------------------
path = Path(__file__)
ROOT_DIR = path.parent.absolute()
config_path = os.path.join(ROOT_DIR, "config.ini")
#------------------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')
train_url = config.get('URLPATH', 'train_url')
test_url = config.get('URLPATH', 'test_url')

def data_preprocess(data, type: str):
    
    """
    A function to retrieve a data from a  given url 
    and then preprocess to store cleaned label and 
    train dataset into our data directory """

    label = []
    features = []

    with open(data, 'r') as f:
        lines = f.readlines()
    
    if type == 'train':
        label = [l[0] for l in lines] # indexing of the first column from each line

        label[(i for i, x in enumerate(label) if x == 0)] = -1  # type: ignore
        columns = [re.sub(r'[^\w]', ' ',l[1:]).split() for l in lines]
    else:
        label = []
        columns = [re.sub(r'[^\w]', ' ',l).split() for l in lines]
    for col in columns:
        lines = [0]*100000
        for idx, val in enumerate(col):
            lines[int(val)] = 1
        features.append(lines)
    return features, label
if __name__ == '__main__':
    # train_data = pd.read_csv(train_url, sep=',')
    # test_data = pd.read_csv(test_url, sep=',')
    train_features, label = data_preprocess(train_url, 'train')
    # test_features, label = data_preprocess(test_data, 'test')
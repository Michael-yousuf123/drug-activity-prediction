import re
import os
from pathlib import Path
import configparser
#------------------------------------------------------------
path = Path(__file__)
ROOT_DIR = path.parent.absolute()
config_path = os.path.join(ROOT_DIR, "config.ini")
#------------------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')
train_path = config.get('URLPATH', 'train_path')
test_path = config.get('URLPATH', 'test_path')

def data_preprocess(data, type: str):
    
    """
    A function to retrieve a data from a  given url 
    and then preprocess to store cleaned label and 
    train dataset into our data directory """

    label = []
    features = []

    with open(data, 'r') as f:
        lines = f.readlines()
    if type == "train":
        labels = [int(l[0]) for l in lines]
        for index, item in enumerate(labels):
            if (item == 0):
                labels[index] = -1
        columns = [re.sub(r'[^\w]', ' ',l[1:]).split() for l in lines]
         
    else:
        label = []
        columns = [re.sub(r'[^\w]', ' ',l).split() for l in lines]
    for col in columns:
        lines = [0]*100001
        for idx, val in enumerate(col):
            lines[int(val)] = 1
        features.append(lines)
    return features, label
if __name__ == '__main__':
    train_features, label = data_preprocess(train_path, 'train')
    test_features, label = data_preprocess(test_path, "test")
    # test_features, label = data_preprocess(test_data, 'test')
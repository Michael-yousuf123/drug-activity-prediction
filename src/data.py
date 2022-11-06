import re
import numpy as np 
import pandas as pd 

def data_preprocess(url: str = None, type: str =None):
    
    """
    A function to retrieve a data from a  given url 
    and then preprocess to store cleaned label and 
    train dataset into our data directory """

    label = []
    features = []

    with open(url, 'r') as f:
        lines = f.readlines()
    
    if type == 'train':
        label = [l[0] for l in lines] # indexing of the first column from each line
        label[[i for i, x in enumerate(label) if x == 0]] = -1
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
    data_preprocess()
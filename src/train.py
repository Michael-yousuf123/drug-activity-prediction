import os
from pathlib import Path
import configparser
import argparse
import data
from dimension import *
import model_dispatch
from dimension import *
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

# import dataset 
path = Path(__file__)
ROOT_DIR = path.parent.absolute()
config_path = os.path.join(ROOT_DIR, "config.ini")
#------------------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')



def train():
    """
    
    """

    train_path = config.get('URLPATH', 'train_path')
    test_path = config.get('URLPATH', 'test_path')
    features, labels = data.data_preprocess(train_path, "train")
    test_features, test_labels = data.data_preprocess(test_path, "test")

    # do dimensionality reduction
    dim_types = ["pca", "lda"]
    K = 100

    dr = DimensionalityReduction(dim_types, K)
    reduced_feat = dr.fit_transform(features, labels)

    test_features, test_labels = data.data_preprocess(test_path, "test")
    test_reduced_features = dr.transform(test_features)

    # fetch the model from model_dispatcher
    clf = model_dispatch.models[model]
    # fit the model on training data
    clf.fit(reduced_feat, labels)
    # create predictions for validation samples
    preds = clf.predict(test_reduced_features)
    # calculate & print accuracy
    accuracy = metrics.f1_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

if __name__ == '__main__':
import os
from pathlib import Path
import configparser
from src.data import *
from dimension import *
from src.model_dispatch import models
from dimension import *
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, auc

k_fold = 10

def training(fold, ):
    """
    """
        # fetch the model from model_dispatcher
    clf = model_dispatch.models[model]
    # fir the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")





if __name__ == '__main__':

    # import dataset 
    path = Path(__file__)
    ROOT_DIR = path.parent.absolute()
    config_path = os.path.join(ROOT_DIR, "config.ini")
    #------------------------------------------------------------
    config = configparser.ConfigParser()
    config.read('config.ini')
    train_path = config.get('URLPATH', 'train_path')
    test_path = config.get('URLPATH', 'test_path')
    features, labels = data.data_preprocess(train_path, "train")
    test_features, labels = data.data_preprocess(test_path, "test")

    # do dimensionality reduction

    dim_types = ["pca", "lda"]
    K = 100

    dr = DimensionalityReduction(dim_types, K)
    reduced_feat = dr.fit_transform(features, labels)
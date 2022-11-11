import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class DimensionalityReduction(BaseEstimator, TransformerMixin):
    """class to compute dimensionality reduction"""

    def __init__(self, types, K):
        """function to initialize the instance 
        attributes of the class 
        ------------
        Parameter
        ------------
        types: types of dimensionality reduction
        N: Number of neighbors if type is non linear
        K: Number of components"""
        self.pca = None
        self.lda = None
        self.K = K
        for idx, item in enumerate(types):
            if "pca" in item:
                self.pca = PCA(K[idx])
            if "lda" in item:
                self.lda = LinearDiscriminantAnalysis()
    
    def fit(self, X, y = None):
        """
        """
        if self.pca is not None:
            self.pca.fit(X)
        if self.lda is not None:
            self.lda.fit(X, y)      
    
    def transform(self, X):
        """
        """
        if self.pca is not None:
            np.array(self.pca.transform(X))
        if self.lda is not None:
            np.array(self.lda.transform(X))
        
    def fit_transform(self, X, y=None):
        """
        """
        self.fit(X, y)
        return self.transform(X)
if __name__ == '__main__':

    X = [[1, 2, 3], [3, 4, 5], [-1, 4, 1], [-1, -5, 3], [-3, 4, 0] ,[3, -5, -4]]
    y = [0, 1, 2, 1, 0, 2]

    # edr = DimensionalityReduction(["pca"], [2])
    # edr.fit(X, y)
    # print(edr.transform(X))
     
    edr = DimensionalityReduction(["pca", "lda"], [2, 2])
    print(edr.fit_transform(X,y))

    edr = DimensionalityReduction(["pca", "lda"], [2, 2])

    print(edr.fit_transform(X,y))
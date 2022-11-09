import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import manifold

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
        self._types = types
        self.K = K
        for idx, item in enumerate(types):
            if "pca" in item:
                self.pca = PCA(K[idx])
            if "lda" in item:
                self.lda = LinearDiscriminantAnalysis()
            if "isomap" in item:
                self.isomap = manifold.Isomap()
            if "lle" in item:
                self.lle = manifold.LocallyLinearEmbedding()
    
    def fit(self, X, y = None):
        """
        """
        if self.pca is not None:
            self.pca.fit(X)
        if self.lda is not None:
            self.lda.fit(X, y)
        if self.isomap is not None:
            self.isomap.fit(X)
        if self.lle is not None:
            self.lle.fit(X)       
    
    def transform(self, X):
        """
        """
        if self.pca is not None:
            np.array(self.pca.transform(X))
        if self.lda is not None:
            np.array(self.lda.transform(X))
        if self.isomap is not None:
            np.array(self.isomap.transform(X))
        if self.pca is not None:
            np.array(self.lle.transform(X))
        
    def fit_transform(self, X, y=None):
        """
        """
        self.fit(X, y)
        return self.transform(X)

    # edr = DimensionalityReduction(["pca", "lda"], [2, 2])

    # print edr.fit_transform(X,y)
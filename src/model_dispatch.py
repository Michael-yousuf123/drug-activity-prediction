from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn import ensemble
from sklearn import tree

# models = [('dt', DecisionTreeClassifier()),
#         ('bn', BernoulliNB()),
#         ('rf', RandomForestClassifier()),
#         ('AdaBoost', AdaBoostClassifier()),
#         ('GBM', GradientBoostingClassifier())]

models = {

        "dt": tree.DecisionTreeClassifier(
        criterion="entropy"),
        "bn": BernoulliNB(),
        "ab": ensemble.AdaBoostClassifier(),
        "gb": ensemble.GradientBoostingClassifier(),
        "rf": ensemble.RandomForestClassifier(),
}

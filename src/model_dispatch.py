from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
from sklearn import tree
from xgboost import XGBClassifier

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
        "xgb": XGBClassifier(n_jobs=-1, max_depth=7, n_estimators=200)
}

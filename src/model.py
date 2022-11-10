from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

models = [('dt', DecisionTreeClassifier()),
        ('bn', BernoulliNB()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('AdaBoost', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier())]

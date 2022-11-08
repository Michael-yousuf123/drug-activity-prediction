from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB


models = [('dt', DecisionTreeClassifier()),
        ('bn', BernoulliNB()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())]
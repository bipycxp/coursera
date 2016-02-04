import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv(
    'titanic.csv',
    index_col='PassengerId',
    usecols=['PassengerId', 'Survived', 'Pclass', 'Fare', 'Age', 'Sex']
)

data = data.dropna()

data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

clf = DecisionTreeClassifier(random_state=241)

features = ['Pclass', 'Fare', 'Age', 'Sex']

clf.fit(pandas.DataFrame(data, columns=features), data['Survived'])

feature_values = sorted(zip(features, clf.feature_importances_), key=lambda item: item[1], reverse=True)

print 'Two best features: %s %s' % (feature_values[0][0], feature_values[1][0])

import pandas
import re
from scipy.stats.stats import pearsonr

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

count = data['Sex'].count()

sexCount = data['Sex'].value_counts()
survivedCount = data['Survived'].value_counts()
classCount = data['Pclass'].value_counts()

names = data['Name']

females = []

for name in names:
    search = re.search('(?<=Miss. )\w+|Mrs.[ \w]+\((\w+)', name)
    if search:
        females.append(search.group(0))

females = pandas.Series(females)

bestName = females.value_counts().keys()[0]

print '1. Males and Females: %i %i' % (sexCount['male'], sexCount['female'])
print '2. Survived percent: %.2f' % (float(survivedCount[1] * 100) / count)
print '3. First class percent: %.2f' % (float(classCount[1] * 100) / count)
print '4. Average and median age: %.2f %i' % (data['Age'].mean(), data['Age'].median())
print '5. Pearson: %.2f' % (pearsonr(data['SibSp'], data['Parch'])[0])
print '6. Best female name: %s' % bestName

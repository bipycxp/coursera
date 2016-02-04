import sklearn.datasets
import sklearn.cross_validation
import sklearn.neighbors
import sklearn.preprocessing

import numpy


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

data = sklearn.datasets.load_boston()

X = sklearn.preprocessing.scale(data['data'])
y = data['target']

KFold = sklearn.cross_validation.KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)

bestP = 1
bestScore = 0

for p in numpy.linspace(1, 10, 200):
    regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    cl = sklearn.cross_validation.cross_val_score(regressor.fit(X, y), cv=KFold, scoring='mean_squared_error', X=X, y=y)

    score = cl.max()

    if p == 1:
        bestScore = score
    elif bestScore < score:
        bestScore = score
        bestP = p

    print 'p = %f, score = %.2f' % (p, score)

print '1. Best p = %.2f, score = %f' % (bestP, bestScore)

create_answer_file(1, round(bestP, 2))

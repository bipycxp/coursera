import pandas
import sklearn.cross_validation
import sklearn.neighbors
import sklearn.preprocessing


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

data = pandas.read_csv('wine.csv', header=None)

X = data.iloc[:, 1:]

y = data.iloc[:, 0]

KFold = sklearn.cross_validation.KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)

for i in range(1, 3):
    bestScore = 0
    bestK = 0

    for k in range(1, 51):
        if i == 2:
            X = sklearn.preprocessing.scale(X)

        classifier = sklearn.neighbors.KNeighborsClassifier(k)
        cl = sklearn.cross_validation.cross_val_score(classifier.fit(X, y), cv=KFold, scoring='accuracy', X=X, y=y)

        score = cl.mean()

        if bestScore < score:
            bestScore = score
            bestK = k

    print '%i. Best k = %i, score = %.2f' % (i, bestK, bestScore)

    create_answer_file(i * 2 - 1, bestK)
    create_answer_file(i * 2, round(bestScore, 2))

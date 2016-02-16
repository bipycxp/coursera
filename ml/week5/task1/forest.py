from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

data = read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

kf = KFold(len(data), 5, True, 1)

min_estimators_count = 0

for i in range(1, 51):
    score = []

    for train_index, test_index in kf:
        X_train, X_test = data.ix[train_index, :-1], data.ix[test_index, :-1]
        y_train, y_test = data.Rings[train_index], data.Rings[test_index]

        rf = RandomForestRegressor(n_estimators=i, random_state=1)
        rf.fit(X_train, y_train)

        score.append(r2_score(y_test, rf.predict(X_test)))

    if sum(score) / float(len(score)) > 0.52:
        min_estimators_count = i

        break

print '1. Min trees count: %i' % min_estimators_count

create_answer_file(1, min_estimators_count)

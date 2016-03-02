import pandas
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from time import time
from sklearn.preprocessing import StandardScaler
from numpy import arange
from sklearn.linear_model import LogisticRegression
import numpy as np

features = pandas.read_csv('features.csv', index_col='match_id')

features.head()

y = features.radiant_win.values

features.drop([
    'duration',
    'radiant_win',
    'tower_status_radiant',
    'tower_status_dire',
    'barracks_status_radiant',
    'barracks_status_dire'
], axis=1, inplace=True)

print 'Missed features:'

for field, total in features.count().iteritems():
    if total != len(features):
        print '%s - %d' % (field, total)

features.fillna(0, inplace=True)

X = features.ix[:, :]

kfold = KFold(n=len(features), n_folds=5, shuffle=True)

#1 Gradient boosting

print '\nGradient boosting:'

for trees_count in [10, 20, 30]:
    t = time()

    estimator = GradientBoostingClassifier(n_estimators=trees_count)
    score = cross_val_score(estimator=estimator, X=X, y=y, scoring='roc_auc', cv=kfold).mean()

    t = time() - t

    print 'Trees: %i, score: %.4f, time: %.2f s' % (trees_count, score, t)

#2 Logistic Regression

scaler = StandardScaler().fit(features)

C_arr = [10 ** x for x in arange(-3, 3, 1)]

print '\nLogistic regression with heroes:'

best_score = 0
best_c = 0

for C in C_arr:
    score = cross_val_score(
        LogisticRegression(C=C),
        X=scaler.transform(X),
        y=y,
        scoring='roc_auc',
        cv=kfold
    ).mean()

    print 'Score: %.4f, C: %.4f' % (score, C)

    if score > best_score:
        best_score = score
        best_c = C

print 'Best score: %.4f, C: %.4f' % (best_score, best_c)

new_features = pandas.read_csv('features.csv', index_col='match_id')

new_features.drop([
    'duration',
    'radiant_win',
    'tower_status_radiant',
    'tower_status_dire',
    'barracks_status_radiant',
    'barracks_status_dire',
    'lobby_type',
    'r1_hero',
    'r2_hero',
    'r3_hero',
    'r4_hero',
    'r5_hero',
    'd1_hero',
    'd2_hero',
    'd3_hero',
    'd4_hero',
    'd5_hero'
], axis=1, inplace=True)

new_features.fillna(0, inplace=True)

new_X = new_features.ix[:, :]

scaler = StandardScaler().fit(new_features)

print '\nLogistic regression without heroes:'

best_score = 0
best_c = 0

for C in C_arr:
    score = cross_val_score(
        LogisticRegression(C=C),
        X=scaler.transform(new_X),
        y=y,
        scoring='roc_auc',
        cv=kfold
    ).mean()

    print 'Score: %.4f, C: %.4f' % (score, C)

    if score > best_score:
        best_score = score
        best_c = C

print 'Best score: %.4f, C: %.4f' % (best_score, best_c)

heroes = pandas.Series()

for h in ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']:
    heroes = heroes.append(features[h])

print '\nUnique heroes: %i' % len(heroes.unique())

N = heroes.max()
X_pick = np.zeros((features.shape[0], N))

for i, match_id in enumerate(features.index):
    for p in xrange(5):
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

print '\nLogistic regression with "hero pack":'

best_score = 0
best_c = 0

for C in C_arr:
    score = cross_val_score(
        LogisticRegression(C=C),
        X=np.hstack([scaler.transform(new_X), X_pick]),
        y=y,
        scoring='roc_auc',
        cv=kfold
    ).mean()

    print 'Score: %.4f, C: %.4f' % (score, C)

    if score > best_score:
        best_score = score
        best_c = C

print 'Best score: %.4f, C: %.4f' % (best_score, best_c)

features_test = pandas.read_csv('features_test.csv', index_col='match_id')

features_test.fillna(0, inplace=True)

X_pick_test = np.zeros((features_test.shape[0], N))

for i, match_id in enumerate(features_test.index):
    for p in xrange(5):
        X_pick_test[i, features_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, features_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

features_test.drop([
    'lobby_type',
    'r1_hero',
    'r2_hero',
    'r3_hero',
    'r4_hero',
    'r5_hero',
    'd1_hero',
    'd2_hero',
    'd3_hero',
    'd4_hero',
    'd5_hero'
], axis=1, inplace=True)

scaler = StandardScaler().fit(new_features)

X_test = features_test.ix[:, :]

logisticRegression = LogisticRegression(C=0.1)
logisticRegression.fit(np.hstack([scaler.transform(new_features), X_pick]), y)

scaler = StandardScaler().fit(features_test)

results = pandas.DataFrame(
    index=features_test.index,
    data=logisticRegression.predict_proba(np.hstack((scaler.transform(features_test), X_pick_test)))[:, 1],
    columns=['radiant_win']
)

results.to_csv('predictions.csv')

print '\nMin and max predicts: %.4f %.4f' % (np.min(results), np.max(results))

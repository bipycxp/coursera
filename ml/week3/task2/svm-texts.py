import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

X = newsgroups.data
y = newsgroups.target

Tfid = TfidfVectorizer()
X = Tfid.fit_transform(X, y)

cv = cross_validation.KFold(n=len(y), n_folds=5, random_state=241)
clf = SVC(kernel='linear', random_state=241)

grid = {'C': np.power(10.0, np.arange(-5, 6))}

gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

bestSvc = gs.best_estimator_

indexes = np.argsort(np.absolute(np.asarray(bestSvc.coef_.todense()).reshape(-1)))[-10:]

words = []

for i in indexes:
    words.append(Tfid.get_feature_names()[i])

words.sort()

answer = ' '.join(str(w) for w in words)

print '10 words: %s' % answer

create_answer_file(1, answer)

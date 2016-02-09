import pandas
from sklearn.svm import SVC


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

data = pandas.read_csv('svm-data.csv', header=None)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

clf = SVC(C=100000, random_state=241)
clf.fit(X, y)

support_numbers = list(clf.support_ + 1)

support_numbers.sort()

answer = ' '.join(str(n) for n in support_numbers)

print 'Support numbers: %s' % answer

create_answer_file(1, answer)

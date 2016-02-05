import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

testData = pandas.read_csv('perceptron-test.csv', header=None)
trainData = pandas.read_csv('perceptron-train.csv', header=None)

X_test = testData.iloc[:, 1:]
y_test = testData.iloc[:, 0]

X_train = trainData.iloc[:, 1:]
y_train = trainData.iloc[:, 0]

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

first_score = accuracy_score(y_test, clf.predict(X_test))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, y_train)

second_score = accuracy_score(y_test, clf.predict(X_test_scaled))

answer = second_score - first_score

print 'Second score - First score = %.3f' % answer

create_answer_file(1, round(answer, 3))

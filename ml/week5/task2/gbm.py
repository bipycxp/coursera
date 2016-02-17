from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from math import exp
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()


def sigmoid(predicts):
    return [1 / (1 + exp(-p)) for p in predicts]

data = read_csv('gbm-data.csv')

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

min_test_loss_index = 0

for r in [1, 0.5, 0.3, 0.2, 0.1]:
    train_loss, test_loss = [], []

    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=r)
    clf.fit(X_train, y_train)

    for predict in clf.staged_decision_function(X_train):
        train_loss.append(log_loss(y_train, sigmoid(predict)))

    for predict in clf.staged_decision_function(X_test):
        test_loss.append(log_loss(y_test, sigmoid(predict)))

    plt.figure()
    plt.plot(train_loss, 'g', linewidth=2, label="Train")
    plt.plot(test_loss, 'r', linewidth=2, label="Test")
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Train and test loss for learning rate = %.1f' % r)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('learning_rate_%.1f.png' % r)
    plt.show()

    if r == 0.2:
        min_test_loss = min(test_loss)
        min_test_loss_index = test_loss.index(min_test_loss)

        print '2. Min log-loss score and index for learning_rate = 0.2: %.2f %i' % (min_test_loss, min_test_loss_index)

        create_answer_file(2, '%.2f %i' % (min_test_loss, min_test_loss_index))

answer = 'overfitting'

print '1. Clf is %s' % answer

create_answer_file(1, answer)

clf = GradientBoostingClassifier(n_estimators=min_test_loss_index, verbose=True, random_state=241)
clf.fit(X_train, y_train)

predict = clf.predict_proba(X_test)
test_loss = log_loss(y_test, predict)

print '3. Log loss score = %.2f' % test_loss

create_answer_file(3, '%.2f' % test_loss)

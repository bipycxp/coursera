from pandas import read_csv
from math import exp, sqrt
from sklearn.metrics import roc_auc_score


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()


def calc_w(x, y, w, C, k):
    s = 0
    l = len(x)

    result = []

    for n in range(0, 2):
        for i in range(0, l):
            s += y[i] * x[i][n] * (1 - 1 / (1 + exp(-y[i] * (w[0] * x[i][0] + w[1] * x[i][1]))))

        result.append(w[n] + k * s / l - k * C * w[n])
        s = 0

    return result


def grad(x, y, w, C=0, k=0.1):
    w_prev = []

    for i in range(0, 10000):
        w = calc_w(x, y, w=w, C=C, k=k)

        if i == 0 or sqrt((w[0] - w_prev[0]) ** 2 + (w[1] - w_prev[1]) ** 2) > 1e-5:
            w_prev = w

            continue
        else:
            break

    return w


def calc_probabilities(x, w):
    prs = []

    for i in range(0, len(x)):
        prs.append(1 / (1 + exp(-w[0] * x[i][0] - w[1] * x[i][1])))

    return prs


data = read_csv('data-logistic.csv', header=None)

x = data.iloc[:, 1:].values.tolist()
y = data.iloc[:, 0].values.tolist()

w_vector = grad(x, y, w=[0, 0])
w_vector_c = grad(x, y, w=[0, 0], C=10)

pr = calc_probabilities(x, w_vector)
pr_c = calc_probabilities(x, w_vector_c)

auc_roc = roc_auc_score(y, pr)
auc_roc_c = roc_auc_score(y, pr_c)

print 'AUC-ROC without reg: %.3f, with reg: %.3f' % (auc_roc, auc_roc_c)

create_answer_file(1, '%.3f %.3f' % (auc_roc, auc_roc_c))

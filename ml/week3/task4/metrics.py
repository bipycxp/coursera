from pandas import read_csv
import sklearn.metrics as metrics


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

data = read_csv('classification.csv')

TP_count = 0
FP_count = 0
FN_count = 0
TN_count = 0

for i in range(len(data)):
    row = data.iloc[i]

    if row['true'] == 1 and row['pred'] == 1:
        TP_count += 1
    elif row['true'] == 0 and row['pred'] == 1:
        FP_count += 1
    elif row['true'] == 1 and row['pred'] == 0:
        FN_count += 1
    elif row['true'] == 0 and row['pred'] == 0:
        TN_count += 1

print '1. TP: %i, FP: %i, FN: %i, TN: %i' % (TP_count, FP_count, FN_count, TN_count)

create_answer_file(1, '%i %i %i %i' % (TP_count, FP_count, FN_count, TN_count))

accuracy_score = round(metrics.accuracy_score(data['true'], data['pred']), 2)
precision_score = round(metrics.precision_score(data['true'], data['pred']), 2)
recall_score = round(metrics.recall_score(data['true'], data['pred']), 2)
f1_score = round(metrics.f1_score(data['true'], data['pred']), 2)

print '2. Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1: %.2f' % (accuracy_score, precision_score, recall_score, f1_score)

create_answer_file(2, '%.2f %.2f %.2f %.2f' % (accuracy_score, precision_score, recall_score, f1_score))

data_2 = read_csv('scores.csv')

score_names = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']

top_score_name = ''
top_score = 0

for c in score_names:
    score = metrics.roc_auc_score(data_2['true'], data_2[c])

    if score > top_score:
        top_score_name = c
        top_score = score

print '3. Top AUC-ROC is "%s"' % top_score_name

create_answer_file(3, '%s' % top_score_name)

top_precision_name = ''
top_precision = 0

for c in score_names:
    precision_recall_curve = metrics.precision_recall_curve(data_2['true'], data_2[c])

    for i in range(len(precision_recall_curve[2])):
        precision = precision_recall_curve[0][i]
        recall = precision_recall_curve[1][i]

        if recall >= 0.7 and precision > top_precision:
            top_precision_name = c
            top_precision = precision

print '4. Top Precision is "%s"' % top_precision_name

create_answer_file(4, '%s' % top_precision_name)

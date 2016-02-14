from pandas import read_csv
from sklearn.decomposition import PCA
from numpy import corrcoef


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

close_prices = read_csv('close_prices.csv')
djia_index = read_csv('djia_index.csv')

X_prices = close_prices.iloc[:, 1:]
X_index = close_prices.iloc[:, 1:]

pca = PCA(10)
pca.fit(X_prices)

variance_ratio = 0
n = 0

for r in pca.explained_variance_ratio_:
    variance_ratio += r
    n += 1

    if variance_ratio >= 0.9:
        break

print '1. Need %i components to 90%% variance.' % n

create_answer_file(1, n)

components = pca.transform(X_prices)

cor = corrcoef(components[:, 0], djia_index['^DJI'])[0][1]

print '2. Pearson correlation coefficient: %.2f' % cor

create_answer_file(2, round(cor, 2))

max_weight = 0
max_name = ''

for c in range(len(X_prices.columns)):
    weight = pca.components_[0].tolist()[c]

    if weight > max_weight:
        max_weight = weight
        max_name = X_prices.columns.values[c]

print '3. Company with max weight: %s' % max_name

create_answer_file(3, max_name)
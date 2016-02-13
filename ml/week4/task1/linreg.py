from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
import re


def create_answer_file(task, answer):
    f = open('%s.txt' % str(task), 'w')
    f.write(str(answer))
    f.close()

train_data = read_csv('salary-train.csv')
test_data = read_csv('salary-test-mini.csv')

train_data['FullDescription'] = train_data['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))
test_data['FullDescription'] = test_data['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))

train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)
test_data['LocationNormalized'].fillna('nan', inplace=True)
test_data['ContractTime'].fillna('nan', inplace=True)

tfid = TfidfVectorizer(min_df=5)
train_fullDescription_tfid = tfid.fit_transform(train_data['FullDescription'])
test_fullDescription_tfid = tfid.transform(test_data['FullDescription'])

enc = DictVectorizer()
train_cat = enc.fit_transform(train_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
test_cat = enc.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([train_fullDescription_tfid, train_cat])
y_train = train_data['SalaryNormalized']

X_test = hstack([test_fullDescription_tfid, test_cat])

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_test = ridge.predict(X_test)

answer = ' '.join(str(round(s, 2)) for s in y_test)

print '1. Salary predictions: %s' % answer

create_answer_file(1, answer)

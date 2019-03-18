from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation
from sklearn import metrics
from classes.dataset import Dataset

# specific categories for testing (faster load time)
categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

# initialize dataset
dataset = Dataset(categories)
dataset.split_train(100)

# feature extraction
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(dataset.train['data'])

# classification
clf = LabelPropagation(kernel='rbf', gamma=0.89).fit(vectors.todense(), dataset.train['target'])
test_vec = vectorizer.transform(dataset.test['data'])

print('----PREDICTIONS----')
pred = clf.predict(test_vec)
print(len(pred))
for i, p in enumerate(pred):
    print(i, ': ', p)

print('f1 score: ', metrics.f1_score(dataset.test['target'], pred, average='macro'))
print('clf score: ', clf.score(test_vec, dataset.test['target']))
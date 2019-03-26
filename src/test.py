from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation
from sklearn import metrics
from src.classes.dataset import Dataset
from src.functions.Preprocess import get_stopwords

# specific categories for testing (faster load time)
categories = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware']
"""
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',
              'talk.religion.misc']
"""

# initialize dataset
dataset = Dataset(categories)
dataset.split_train(100)

# feature extraction
vectorizer = TfidfVectorizer(stop_words=get_stopwords(), max_df=0.5, min_df=10)
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

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.request
from sklearn.semi_supervised import LabelPropagation
from sklearn import metrics

# specific categories for testing (faster load time)
categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

# fetch dataset
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers'),
                                      categories=categories)

newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers'),
                                     categories=categories)

# get stopwords from file
def get_stopwords():
    f = open('./assets/stopwords.txt')  #https://github.com/suzanv/termprofiling/blob/master/stoplist.txt
    x = f.read().split("\n")
    f.close()
    return x


# feature extraction
vectorizer = TfidfVectorizer(stop_words=get_stopwords())
vectors = vectorizer.fit_transform(newsgroups_train.data)

# classification
clf = LabelPropagation(kernel='rbf').fit(vectors.todense(), newsgroups_train.target)
test_vec = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(test_vec)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))
print(clf.score(test_vec, newsgroups_test.target))
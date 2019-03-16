from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation
from sklearn import metrics
import pprint

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
    f = open('../assets/stopwords.txt')  #https://github.com/suzanv/termprofiling/blob/master/stoplist.txt
    x = f.read().split("\n")
    f.close()
    return x


def remove_regex_words():
    i = 0
    print([newsgroups_train.data[0]])
    with open('../assets/stopwords_regex.txt', 'r') as f:
        words = f.read().split("\n")
        while i < len(words):
            j = 0
            while j < len(newsgroups_train.data):
                newsgroups_train.data[j] = re.sub(words[i], '', newsgroups_train.data[j])
                j += 1
            i += 1
    f.close()
    print([newsgroups_train.data[0]])



# feature extraction
vectorizer = TfidfVectorizer(stop_words=get_stopwords())
vectors = vectorizer.fit_transform(newsgroups_train.data)


clf = LabelPropagation(kernel='rbf', gamma=0.89).fit(vectors.todense(), newsgroups_train.target)
test_vec = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(test_vec)
print(clf.score(test_vec, newsgroups_test.target))
print('f1 score: ', metrics.f1_score(newsgroups_test.target, pred, average='macro'))

remove_regex_words()
vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = LabelPropagation(kernel='rbf', gamma=0.89).fit(vectors.todense(), newsgroups_train.target)
test_vec = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(test_vec)
print(clf.score(test_vec, newsgroups_test.target))
print('f1 score: ', metrics.f1_score(newsgroups_test.target, pred, average='macro'))

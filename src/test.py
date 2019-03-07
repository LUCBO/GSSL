from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.request
from sklearn.semi_supervised import LabelPropagation
from sklearn import metrics
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

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


# lemmatize
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


lemmatizer = WordNetLemmatizer()
sentence = "The striped bats are hanging on their feet for best"
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])


# feature extraction
vectorizer = TfidfVectorizer(stop_words=get_stopwords())
lem = WordNetLemmatizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

# classification
clf = LabelPropagation(kernel='rbf', gamma=0.89).fit(vectors.todense(), newsgroups_train.target)
test_vec = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(test_vec)
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'))
print(clf.score(test_vec, newsgroups_test.target))

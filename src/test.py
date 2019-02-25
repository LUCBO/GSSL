from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.request


# specific categories for testing (faster load time)
categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

# fetch dataset
newsgroups_train = fetch_20newsgroups(subset='train',
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

#print(vectors[0]) #Display first vector
print(vectors.toarray()) #Display as vector
print(len(vectorizer.get_feature_names())) #Display feature count

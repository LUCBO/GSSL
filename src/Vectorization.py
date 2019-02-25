import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
count = CountVectorizer()
tf_idf = TfidfVectorizer()
docs = np.array(['The sky is blue', 'So is the sea.', 'The sky is blue and so is the sea.'])
bag = count.fit_transform(docs)
np.set_printoptions(precision=2)
tf_idf_vectorized = tf_idf.fit_transform(docs)
tfNoNorm = TfidfVectorizer(norm='')
tf_vectorizedNoNorm = tfNoNorm.fit_transform(docs)


def bow():
    print('\nBOW')
    print(count.vocabulary_)
    print(bag.toarray())
    print(count.get_feature_names())


def tfidf():
    print('\nTF-IDF')
    print(tf_idf.vocabulary_)
    print(tf_idf_vectorized.toarray())
    print(tf_vectorizedNoNorm.toarray())
    print(tf_idf.get_feature_names())


bow()
tfidf()

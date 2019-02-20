#from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
#categories = ['alt.atheism']
#newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

newsgroups_train = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
print(newsgroups_train)

# turn the text into vectors of numerical values suitable for statistical analysis
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train)
print(vectors)
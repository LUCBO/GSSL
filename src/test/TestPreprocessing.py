import src.main.Preprocess
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism']

# fetch dataset
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers'),
                                     categories=categories)

before = [newsgroups_train.data[0]]
result = src.main.Preprocess.process(newsgroups_train, newsgroups_test)
newsgroups_train = result[0]
newsgroups_test = result[1]
print(before)
print([newsgroups_train.data[0]])

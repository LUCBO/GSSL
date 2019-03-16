import src.main.Preprocess
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism']

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers'),
                                     categories=categories)

before = [newsgroups_train.data[0]]
src.main.Preprocess.process(newsgroups_train, newsgroups_test)  # Only first time

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers'),
                                     categories=categories)

src.main.Preprocess.load_improved_newsgroup(newsgroups_train, newsgroups_test)  # Only thing needed other times
print(before)
print([newsgroups_train.data[0]])

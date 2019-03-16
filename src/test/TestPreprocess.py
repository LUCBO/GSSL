import src.main.Preprocess
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism']

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      )
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     )

before = [newsgroups_train.data[0]]
# src.main.Preprocess.process(newsgroups_train, newsgroups_test)  # Only first time

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      )
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     )

src.main.Preprocess.load_improved_newsgroup(newsgroups_train, newsgroups_test)  # Only thing needed other times
print(before)
print([newsgroups_train.data[0]])

import src.functions.Preprocess
from sklearn.datasets import fetch_20newsgroups
from src.classes.dataset import Dataset

categories = ['alt.atheism']

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      )
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     )

before = [newsgroups_train.data[0]]
# src.functions.Preprocess.process(newsgroups_train, newsgroups_test)  # Only first time

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      )
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     )

dataset = Dataset(categories)
src.functions.Preprocess.load_improved_newsgroup(dataset.train, dataset.test)  # Only thing needed other times
print(before)
print([dataset.train['data'][0]])

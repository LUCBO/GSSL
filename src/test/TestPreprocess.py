import src.functions.Preprocess
from sklearn.datasets import fetch_20newsgroups
from src.classes.dataset import Dataset

categories = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',
              'talk.religion.misc']

category = ['alt.atheism']

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=category)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=category)

before = [newsgroups_train.data[0]]
before_testdata = [newsgroups_test.data[0]]
# src.functions.Preprocess.process(categories)  # Only first time
src.functions.Preprocess.print_v2_docs(categories)
dataset = Dataset(category)
print(before)
print([dataset.train['data'][0]])
print(before_testdata)
print([dataset.test['data'][0]])


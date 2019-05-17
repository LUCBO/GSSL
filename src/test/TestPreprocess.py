import src.functions.Preprocess
from sklearn.datasets import fetch_20newsgroups
from src.classes.dataset import Dataset

# handles the preprocessing process. The categories used for classification must not be processed with those that
# are not.
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

category = ['rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey']

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=category)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=category)
before = [newsgroups_train.data[11]]
before_testdata = [newsgroups_test.data[0]]
dataset = Dataset(category)
src.functions.Preprocess.process(category)  # Only first time
src.functions.Preprocess.print_v2_docs(category)
src.functions.Preprocess.print_v2_test_docs_vocabulary(category)
dataset = Dataset(category)
print(before)
print([dataset.train['data'][11]])
print(before_testdata)
print([dataset.test['data'][0]])

import src.functions.Vocabulary
from src.classes import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
"""
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
"""
categories = ['rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey']

src.functions.Vocabulary.create_vocabulary(categories, 10)
voc = src.functions.Vocabulary.get_vocabulary(categories)
print(len(voc))
dataset = Dataset(categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(dataset.train['data'])
print(len(vectorizer.vocabulary_))
print(src.functions.Vocabulary.get_vocabulary(['rec.autos']))
print(src.functions.Vocabulary.get_vocabulary(categories))
dataset = Dataset(categories)
x1 = len(dataset.test['data'])
dataset.load_preprocessed_vocabulary_in_use(categories)
x2 = len(dataset.test['data'])
print("Before: " + x1.__str__())
print("After: " + x2.__str__())


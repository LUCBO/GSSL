from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from src.classes.dataset import Dataset
from src.functions.Preprocess import get_stopwords
from src.functions.ConfusionMatrix import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import src.functions.Vocabulary as voc


# specific categories for testing (faster load time)
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

categories = ['alt.atheism',
              'comp.graphics',
              'rec.autos',
              'sci.space',
              'talk.politics.guns']


# initialize dataset
dataset = Dataset(categories)
dataset.load_preprocessed_vocabulary_in_use(categories)
dataset.split_train_bayers(100)

# feature extraction
vectorizer = TfidfVectorizer(vocabulary=voc.get_vocabulary(categories))
vectors = vectorizer.fit_transform(dataset.train['data'])

clf = MultinomialNB().fit(vectors.todense(), dataset.train['target'])
test_vec = vectorizer.transform(dataset.test['data'])
pred = clf.predict(test_vec.todense())

print('f1 score Naive Bayes: ', metrics.f1_score(dataset.test['target'], pred, average='macro'))
print('clf score Naive Bayes: ', clf.score(test_vec.todense(), dataset.test['target']))

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred, classes=categories,
                      title='Confusion matrix (Naive Bayes), without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred, classes=categories, normalize=True,
                      title='Normalized confusion matrix (Naive Bayes)')
plt.show()

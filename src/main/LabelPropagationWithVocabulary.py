from sklearn.feature_extraction.text import TfidfVectorizer
from src.classes import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn import metrics
from src.classes import Dataset
from src.functions.Preprocess import get_stopwords
from src.functions.ConfusionMatrix import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
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
dataset.split_train_true(100)
dataset_knn = Dataset(categories)
dataset_knn.split_train_true(100)

# feature extraction
vectorizer = TfidfVectorizer(vocabulary=voc.get_vocabulary(categories))
vectors = vectorizer.fit_transform(dataset.train['data'])
vectors_knn = vectorizer.fit_transform((dataset_knn.train['data']))

# classification
# use max_iter=10 when 20 categories
clf_rbf = LabelPropagation(kernel='rbf').fit(vectors.todense(), dataset.train['target'])
clf_knn = LabelSpreading(kernel='knn', n_neighbors=10).fit(vectors_knn.todense(), dataset_knn.train['target'])
test_vec = vectorizer.transform(dataset.test['data'])
test_vec_knn = vectorizer.transform(dataset_knn.test['data'])

print('----PREDICTIONS----')
pred_rbf = clf_rbf.predict(test_vec.todense())
pred_knn = clf_rbf.predict(test_vec_knn.todense())

print('f1 score rbf: ', metrics.f1_score(dataset.test['target'], pred_rbf, average='macro'))
print('clf score rbf: ', clf_rbf.score(test_vec.todense(), dataset.test['target']))
print('f1 score knn: ', metrics.f1_score(dataset_knn.test['target'], pred_knn, average='macro'))
print('clf score knn: ', clf_knn.score(test_vec_knn.todense(), dataset_knn.test['target']))

np.set_printoptions(precision=2)

"""
# Plot non-normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred_rbf, classes=categories,
                      title='Confusion matrix (RBF), without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred_rbf, classes=categories, normalize=True,
                      title='Normalized confusion matrix (RBF)')
plt.show()
"""

# Plot non-normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred_knn, classes=categories,
                      title='Confusion matrix (KNN), without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred_knn, classes=categories, normalize=True,
                      title='Normalized confusion matrix (KNN)')
plt.show()

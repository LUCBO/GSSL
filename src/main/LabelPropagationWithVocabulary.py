from sklearn.feature_extraction.text import TfidfVectorizer
from src.classes import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn import metrics
from src.classes import Dataset
from src.functions.Preprocess import get_stopwords
from src.functions.Preprocess import print_v2_test_docs_vocabulary_labeled
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
dataset.load_preprocessed(categories)
dataset.split_train_true(100)
print_v2_test_docs_vocabulary_labeled(categories)
dataset.load_preprocessed_test_vocabulary_labeled_in_use(categories)

dataset_knn = Dataset(categories)
dataset_knn.load_preprocessed_vocabulary_in_use(categories)
dataset_knn.split_train_true(100)
print_v2_test_docs_vocabulary_labeled(categories)
dataset_knn.load_preprocessed_test_vocabulary_labeled_in_use(categories)

# feature extraction
vectorizer_rbf = TfidfVectorizer(vocabulary=voc.get_vocabulary_only_labeled(categories))
vectorizer_knn = TfidfVectorizer(vocabulary=voc.get_vocabulary_only_labeled(categories))
vectors = vectorizer_rbf.fit_transform(dataset.train['data'])
vectors_knn = vectorizer_knn.fit_transform((dataset_knn.train['data']))

# classification
# use max_iter=10 when 20 categories
clf_rbf = LabelPropagation(kernel='rbf', gamma=5).fit(vectors.todense(), dataset.train['target'])
clf_knn = LabelSpreading(kernel='knn', n_neighbors=10).fit(vectors_knn.todense(), dataset_knn.train['target'])
test_vec_rbf = vectorizer_rbf.transform(dataset.test['data'])
test_vec_knn = vectorizer_knn.transform(dataset_knn.test['data'])

print('----PREDICTIONS----')
pred_rbf = clf_rbf.predict(test_vec_rbf.todense())
pred_knn = clf_knn.predict(test_vec_knn.todense())

print('f1 score rbf: ', metrics.f1_score(dataset.test['target'], pred_rbf, average='macro'))
print('clf score rbf: ', clf_rbf.score(test_vec_rbf.todense(), dataset.test['target']))
print('f1 score knn: ', metrics.f1_score(dataset_knn.test['target'], pred_knn, average='macro'))
print('clf score knn: ', clf_knn.score(test_vec_knn.todense(), dataset_knn.test['target']))

np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred_rbf, classes=categories,
                      title='Confusion matrix (RBF with vocabulary), without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred_rbf, classes=categories, normalize=True,
                      title='Normalized confusion matrix (RBF with vocabulary)')
plt.show()


# Plot non-normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred_knn, classes=categories,
                      title='Confusion matrix (KNN with vocabulary), without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred_knn, classes=categories, normalize=True,
                      title='Normalized confusion matrix (KNN with vocabulary)')
plt.show()


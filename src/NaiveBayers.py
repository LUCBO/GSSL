from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from src.classes.dataset import Dataset
from src.functions.Preprocess import get_stopwords
from src.functions.ConfusionMatrix import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


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
dataset.split_train_bayers(100)

# feature extraction
vectorizer = TfidfVectorizer(stop_words=get_stopwords(), max_df=0.5, min_df=10)
vectors = vectorizer.fit_transform(dataset.train['data'])

clf = MultinomialNB().fit(vectors.todense(), dataset.train['target'])
test_vec = vectorizer.transform(dataset.test['data'])
pred = clf.predict(test_vec.todense())

mean = np.mean(pred == dataset.test['target'])
print(mean)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred, classes=categories,
                      title='Confusion matrix (KNN), without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(dataset.test['target'], pred, classes=categories, normalize=True,
                      title='Normalized confusion matrix (KNN)')
plt.show()
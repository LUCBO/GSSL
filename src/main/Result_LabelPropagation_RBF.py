from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from src.classes import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn import metrics
from src.classes import Dataset
from src.functions.Preprocess import get_stopwords
from sklearn.naive_bayes import MultinomialNB
import src.functions.Vocabulary as Vocabulary
from src.functions.Preprocess import print_v2_test_docs_vocabulary_labeled

categories = ['alt.atheism',
              'comp.graphics',
              'rec.autos',
              'sci.space',
              'talk.politics.guns']


def run_lp_bow(nbr, str_list, gamma):
    i = 0
    avg_f1 = 0
    avg_accuracy = 0
    while i < 10:
        dataset = Dataset(categories)
        dataset.split_train_true(nbr)
        vectorizer = CountVectorizer(stop_words=get_stopwords(), max_df=0.5, min_df=10)
        vectors = vectorizer.fit_transform(dataset.train['data'])

        clf = LabelPropagation(kernel='rbf', gamma=gamma).fit(vectors.todense(), dataset.train['target'])
        test_vec = vectorizer.transform(dataset.test['data'])
        pred = clf.predict(test_vec.todense())
        avg_f1 += metrics.f1_score(dataset.test['target'], pred, average='macro')
        avg_accuracy += clf.score(test_vec.todense(), dataset.test['target'])
        i += 1
    avg_accuracy = avg_accuracy/10
    avg_f1 = avg_f1/10
    str_list.extend(["RBF BOW Avg f1: " + avg_f1.__str__(), "RBF BOW Avg acc: " + avg_accuracy.__str__()])
    print("Avg f1: " + avg_f1.__str__())
    print("Avg acc: " + avg_accuracy.__str__())


def run_lp_bow_vocabulary(nbr, str_list, gamma):
    i = 0
    avg_f1 = 0
    avg_accuracy = 0
    while i < 10:
        dataset = Dataset(categories)
        dataset.split_train_true(nbr)
        vectorizer = CountVectorizer(vocabulary=Vocabulary.get_vocabulary(categories))
        vectors = vectorizer.fit_transform(dataset.train['data'])

        clf = LabelPropagation(kernel='rbf', gamma=gamma).fit(vectors.todense(), dataset.train['target'])
        test_vec = vectorizer.transform(dataset.test['data'])
        pred = clf.predict(test_vec.todense())
        avg_f1 += metrics.f1_score(dataset.test['target'], pred, average='macro')
        avg_accuracy += clf.score(test_vec.todense(), dataset.test['target'])
        i += 1
    avg_accuracy = avg_accuracy/10
    avg_f1 = avg_f1/10
    str_list.extend(["RBF BOW voc Avg f1: " + avg_f1.__str__(), "RBF BOW voc Avg acc: " + avg_accuracy.__str__()])
    print("Avg f1: " + avg_f1.__str__())
    print("Avg acc: " + avg_accuracy.__str__())


def run_lp_bow_runtime_vocabulary(nbr, str_list, gamma):
    i = 0
    avg_f1 = 0
    avg_accuracy = 0
    while i < 10:
        dataset = Dataset(categories)
        dataset.load_preprocessed(categories)
        dataset.split_train_true(nbr)
        print_v2_test_docs_vocabulary_labeled(categories)
        dataset.load_preprocessed_test_vocabulary_labeled_in_use(categories)
        vectorizer = CountVectorizer(vocabulary=Vocabulary.get_vocabulary(categories))
        vectors = vectorizer.fit_transform(dataset.train['data'])

        clf = LabelPropagation(kernel='rbf', gamma=gamma).fit(vectors.todense(), dataset.train['target'])
        test_vec = vectorizer.transform(dataset.test['data'])
        pred = clf.predict(test_vec.todense())
        avg_f1 += metrics.f1_score(dataset.test['target'], pred, average='macro')
        avg_accuracy += clf.score(test_vec.todense(), dataset.test['target'])
        i += 1
    avg_accuracy = avg_accuracy/10
    avg_f1 = avg_f1/10
    str_list.extend(["RBF BOW runtime voc Avg f1: " + avg_f1.__str__(), "RBF BOW runtime voc Avg acc: "
                     + avg_accuracy.__str__()])
    print("Avg f1: " + avg_f1.__str__())
    print("Avg acc: " + avg_accuracy.__str__())


def run_lp_tfidf(nbr, str_list, gamma):
    i = 0
    avg_f1 = 0
    avg_accuracy = 0
    while i < 10:
        dataset = Dataset(categories)
        dataset.split_train_true(nbr)
        vectorizer = TfidfVectorizer(stop_words=get_stopwords(), max_df=0.5, min_df=10)
        vectors = vectorizer.fit_transform(dataset.train['data'])

        clf = LabelPropagation(kernel='rbf', gamma=gamma).fit(vectors.todense(), dataset.train['target'])
        test_vec = vectorizer.transform(dataset.test['data'])
        pred = clf.predict(test_vec.todense())
        avg_f1 += metrics.f1_score(dataset.test['target'], pred, average='macro')
        avg_accuracy += clf.score(test_vec.todense(), dataset.test['target'])
        i += 1
    avg_accuracy = avg_accuracy/10
    avg_f1 = avg_f1/10
    str_list.extend(["RBF TF-IDF Avg f1: " + avg_f1.__str__(), "RBF TF-IDF Avg acc: " + avg_accuracy.__str__()])
    print("Avg f1: " + avg_f1.__str__())
    print("Avg acc: " + avg_accuracy.__str__())


def run_lp_tfidf_vocabulary(nbr, str_list, gamma):
    i = 0
    avg_f1 = 0
    avg_accuracy = 0
    while i < 10:
        dataset = Dataset(categories)
        dataset.split_train_true(nbr)
        vectorizer = TfidfVectorizer(vocabulary=Vocabulary.get_vocabulary(categories))
        vectors = vectorizer.fit_transform(dataset.train['data'])

        clf = LabelPropagation(kernel='rbf', gamma=gamma).fit(vectors.todense(), dataset.train['target'])
        test_vec = vectorizer.transform(dataset.test['data'])
        pred = clf.predict(test_vec.todense())
        avg_f1 += metrics.f1_score(dataset.test['target'], pred, average='macro')
        avg_accuracy += clf.score(test_vec.todense(), dataset.test['target'])
        i += 1
    avg_accuracy = avg_accuracy/10
    avg_f1 = avg_f1/10
    str_list.extend(["RBF TF-IDF voc Avg f1: " + avg_f1.__str__(), "RBF TF-IDF voc Avg acc: " + avg_accuracy.__str__()])
    print("Avg f1: " + avg_f1.__str__())
    print("Avg acc: " + avg_accuracy.__str__())


def run_lp_tfidf_runtime_vocabulary(nbr, str_list, gamma):
    i = 0
    avg_f1 = 0
    avg_accuracy = 0
    while i < 10:
        dataset = Dataset(categories)
        dataset.load_preprocessed(categories)
        dataset.split_train_true(nbr)
        print_v2_test_docs_vocabulary_labeled(categories)
        dataset.load_preprocessed_test_vocabulary_labeled_in_use(categories)
        vectorizer = TfidfVectorizer(vocabulary=Vocabulary.get_vocabulary(categories))
        vectors = vectorizer.fit_transform(dataset.train['data'])

        clf = LabelPropagation(kernel='rbf', gamma=gamma).fit(vectors.todense(), dataset.train['target'])
        test_vec = vectorizer.transform(dataset.test['data'])
        pred = clf.predict(test_vec.todense())
        avg_f1 += metrics.f1_score(dataset.test['target'], pred, average='macro')
        avg_accuracy += clf.score(test_vec.todense(), dataset.test['target'])
        i += 1
    avg_accuracy = avg_accuracy/10
    avg_f1 = avg_f1/10
    str_list.extend(["RBF TF-IDF runtime voc Avg f1: " + avg_f1.__str__(), "RBF TF-IDF runtime Avg acc: "
                     + avg_accuracy.__str__()])
    print("Avg f1: " + avg_f1.__str__())
    print("Avg acc: " + avg_accuracy.__str__())


# runs all the different preprocessing and feature extraction combinations and prints the result
def get_result(nbr_labeled_docs, gamma):
    str_list = []
    run_lp_bow(nbr_labeled_docs, str_list, gamma)
    run_lp_bow_vocabulary(nbr_labeled_docs, str_list, gamma)
    run_lp_bow_runtime_vocabulary(nbr_labeled_docs, str_list, gamma)
    run_lp_tfidf(nbr_labeled_docs, str_list, gamma)
    run_lp_tfidf_vocabulary(nbr_labeled_docs, str_list, gamma)
    run_lp_tfidf_runtime_vocabulary(nbr_labeled_docs, str_list, gamma)
    x = 0
    while x < len(str_list):
        print(str_list[x])
        x += 1


get_result(10, 5)
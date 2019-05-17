import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
from sklearn.datasets import fetch_20newsgroups
from src.classes import Dataset
from sklearn.feature_extraction.text import CountVectorizer
import src.functions.Vocabulary as voc
from src.classes.dataset import Dataset


# Preprocess the 20 Newsgroups data
def process(categories):
    i = 0
    while i < len(categories):
        trainingdata = fetch_20newsgroups(subset='train',
                                          remove=('headers', 'footers', 'quotes'),
                                          categories=[categories[i]])
        testdata = fetch_20newsgroups(subset='test',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=[categories[i]])

        lemmatize_newsgroup(trainingdata, testdata, categories[i])
        remove_stopwords(trainingdata)
        remove_stopwords(testdata)
        print_docs(trainingdata, testdata, categories[i])
        i += 1
    dataset = Dataset(categories)
    dataset.load_preprocessed_V1(categories)
    remove_frequent_and_infrequent_words(dataset.train)
    print_docs_reduced_feature_count(dataset, categories)
    print_v2_docs(categories)
    print_v2_test_docs_vocabulary(categories)


# Determines how a word shall change to convert it to its base form
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Lemmatization of text in documents
def lemmatize_newsgroup(newsgroups_train, newsgroups_test, category):
    i = 0
    lemmatizer = WordNetLemmatizer()
    size = len(newsgroups_train.data)
    print("Lemmatization in progress...")
    print(category + " training data: ", i, "/", size)
    while i < len(newsgroups_train.data):
        newsgroups_train.data[i] = newsgroups_train.data[i].lower()
        newsgroups_train.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                              nltk.word_tokenize(newsgroups_train.data[i]) if w not in
                                              string.punctuation]))
        i += 1
        print(category + " training data: ", i, "/", size)
    size = len(newsgroups_test.data)
    i = 0
    print(category + " test data: ", i, "/", size)
    while i < len(newsgroups_test.data):
        newsgroups_test.data[i] = newsgroups_test.data[i].lower()
        newsgroups_test.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                             nltk.word_tokenize(newsgroups_test.data[i]) if w not in
                                             string.punctuation]))
        i += 1
        print(category + " test data: ", i, "/", size)
    print("Lemmatization finished")


# prints the training and test documents of a category to a their respective file.
def print_docs(newsgroups_train, newsgroups_test, category):
    i = 0
    print("Printing docs...")
    with open('../assets/20newsgroups/train/newsgroups_train_' + category + '.txt', 'w') as f:
        while i < len(newsgroups_train.data):
            f.write("%s\n" % newsgroups_train.data[i].encode("utf-8"))
            i += 1
        f.close()

    i = 0
    with open('../assets/20newsgroups/test/newsgroups_test_' + category + '.txt', 'w') as f:
        while i < len(newsgroups_test.data):
            f.write("%s\n" % newsgroups_test.data[i].encode("utf-8"))
            i += 1
        f.close()
    print("Printing finished...")


# prints the training documents after frequent and infrequent words have been removed
def print_docs_reduced_feature_count(dataset, categories):
    print("Printing docs...")
    i = 0
    train_category = []
    while i < len(categories):
        train_category.append([])
        i += 1
    i = 0
    while i < len(dataset.train['data']):
        c = 0
        category = dataset.train['target_names'][dataset.train['target'][i]]
        while c < len(categories):
            if category == categories[c]:
                train_category[c].append(dataset.train['data'][i])
                break
            c += 1
        i += 1
    i = 0
    print("Docs sorted")
    while i < len(train_category):
        if len(train_category[i]) > 1:
            with open('../assets/20newsgroups/train/newsgroups_train_' + categories[i] + '.txt', 'w') as f:
                j = 0
                while j < len(train_category[i]):
                    f.write("%s\n" % train_category[i][j])
                    j += 1
                f.close()
        i += 1
    print("Printing finished...")


# prints a new version of the previously printed documents to a new file
# documents without content are not printed
def print_v2_docs(categories):
    i = 0
    removed_train = 0
    removed_test = 0
    print("Printing docs...")
    while i < len(categories):
        with open('../assets/20newsgroups/train2/newsgroups_train_' + categories[i] + '.txt', 'w') as f:
            lines = [line.rstrip('\n') for line in open('../assets/20newsgroups/train/newsgroups_train_'
                                                        + categories[i] + '.txt')]
            j = 0
            while j < len(lines):
                lines[j] = re.sub(r'[^\w]', " ", lines[j])
                lines[j] = re.sub(r'\b[a-zA-Z]\b', " ", lines[j])
                lines[j] = re.sub(r'[ \t]+', " ", lines[j])  # remove extra space or tab
                lines[j] = lines[j].strip() + "\n"
                size = len(lines[j])
                # lines[j] = lines[j][1:size]
                if len(lines[j]) > 4:
                    f.write(lines[j])
                else:
                    removed_train += 1
                j += 1
            f.close()
        with open('../assets/20newsgroups/test2/newsgroups_test_' + categories[i] + '.txt', 'w') as f:
            lines = [line.rstrip('\n') for line in open('../assets/20newsgroups/test/newsgroups_test_'
                                                        + categories[i] + '.txt')]
            j = 0
            dataset = Dataset(categories)
            vectorizer = CountVectorizer(stop_words=get_stopwords(), max_df=0.5, min_df=10)
            vectors = vectorizer.fit_transform(dataset.train['data'])
            vocabulary = vectorizer.vocabulary_
            while j < len(lines):
                lines[j] = re.sub(r'[^\w]', " ", lines[j])
                lines[j] = re.sub(r'\b[a-zA-Z]\b', " ", lines[j])
                lines[j] = re.sub(r'[ \t]+', " ", lines[j])  # remove extra space or tab
                lines[j] = lines[j].strip() + "\n"
                remove_doc = 1
                words = lines[j].split()
                for word in words:
                    if word in vocabulary.keys():
                        remove_doc = 0
                        break
                size = len(lines[j])
                # lines[j] = lines[j][1:size]
                if len(lines[j]) > 4 and not remove_doc:
                    f.write(lines[j])
                else:
                    removed_test += 1
                j += 1
            f.close()
        i += 1
    print("Printing finished")
    print("Removed training doc:", removed_train)
    print("Removed testing doc:", removed_test)


# same function as print_v2_docs but prints a new version of test docs which for when the vocabulary constructed
# using all the training documents are in use
def print_v2_test_docs_vocabulary(categories):
    i = 0
    removed_test = 0
    print("Printing docs...")
    while i < len(categories):
        with open('../assets/20newsgroups/test2vocabulary/newsgroups_test_' + categories[i] + '.txt', 'w') as f:
            lines = [line.rstrip('\n') for line in open('../assets/20newsgroups/test/newsgroups_test_'
                                                        + categories[i] + '.txt')]
            j = 0
            dataset = Dataset(categories)
            vectorizer = CountVectorizer(vocabulary=voc.get_vocabulary(categories))
            vectors = vectorizer.fit_transform(dataset.train['data'])
            vocabulary = vectorizer.vocabulary_
            while j < len(lines):
                lines[j] = re.sub(r'[^\w]', " ", lines[j])
                lines[j] = re.sub(r'\b[a-zA-Z]\b', " ", lines[j])
                lines[j] = re.sub(r'[ \t]+', " ", lines[j])  # remove extra space or tab
                lines[j] = lines[j].strip() + "\n"
                remove_doc = 1
                words = lines[j].split()
                for word in words:
                    if word in vocabulary.keys():
                        remove_doc = 0
                        break
                size = len(lines[j])
                # lines[j] = lines[j][1:size]
                if len(lines[j]) > 4 and not remove_doc:
                    f.write(lines[j])
                else:
                    removed_test += 1
                j += 1
            f.close()
        i += 1
    print("Printing finished")
    print("Removed testing doc:", removed_test)


# same as print_v2_test_docs_vocabulary but for when the runtime vocabulary are in use
def print_v2_test_docs_vocabulary_labeled(categories):
    i = 0
    removed_test = 0
    print("Printing docs...")
    while i < len(categories):
        with open('../assets/20newsgroups/test2vocabulary_labeled/newsgroups_test_' + categories[i] + '.txt', 'w') as f:
            lines = [line.rstrip('\n') for line in open('../assets/20newsgroups/test/newsgroups_test_'
                                                        + categories[i] + '.txt')]
            j = 0
            dataset = Dataset(categories)
            vectorizer = CountVectorizer(vocabulary=voc.get_vocabulary_only_labeled(categories))
            vectors = vectorizer.fit_transform(dataset.train['data'])
            vocabulary = vectorizer.vocabulary_
            while j < len(lines):
                lines[j] = re.sub(r'[^\w]', " ", lines[j])
                lines[j] = re.sub(r'\b[a-zA-Z]\b', " ", lines[j])
                lines[j] = re.sub(r'[ \t]+', " ", lines[j])  # remove extra space or tab
                lines[j] = lines[j].strip() + "\n"
                remove_doc = 1
                words = lines[j].split()
                for word in words:
                    if word in vocabulary.keys():
                        remove_doc = 0
                        break
                size = len(lines[j])
                # lines[j] = lines[j][1:size]
                if len(lines[j]) > 4 and not remove_doc:
                    f.write(lines[j])
                else:
                    removed_test += 1
                j += 1
            f.close()
        i += 1
    print("Printing finished")
    print("Removed testing doc:", removed_test)


# get stopwords from file
def get_stopwords():
    f = open('../assets/stopwords.txt')  # https://github.com/suzanv/termprofiling/blob/master/stoplist.txt
    x = f.read().split("\n")
    f.close()
    return x


# removes words with which occur in less than 10 document and more than 50%
def remove_frequent_and_infrequent_words(newsgroup):
    vectorizer = CountVectorizer(max_df=0.5, min_df=10)
    vectors = vectorizer.fit_transform(newsgroup['data'])
    vocabulary = voc.get_top_n_words(newsgroup['data'], len(vectorizer.vocabulary_))
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(newsgroup['data'])
    vocabulary_with_freq_and_infreq = voc.get_top_n_words(newsgroup['data'], len(vectorizer.vocabulary_))
    i = 0
    while i < len(vocabulary_with_freq_and_infreq):
        vocabulary_with_freq_and_infreq[i] = vocabulary_with_freq_and_infreq[i][0]
        if i < len(vocabulary):
            vocabulary[i] = vocabulary[i][0]
        i += 1
    print(len(vocabulary))
    print(len(vocabulary_with_freq_and_infreq))
    i = 0
    while i < len(vocabulary_with_freq_and_infreq):
        j = 0
        if vocabulary_with_freq_and_infreq[i] not in vocabulary:
            while j < len(newsgroup['data']):
                newsgroup['data'][j] = re.sub(r'\b' + vocabulary_with_freq_and_infreq[i] + '\s', ' ',
                                              newsgroup['data'][j])
                j += 1
        i += 1
        print("Freq/Infreq: ", i, "/", len(vocabulary_with_freq_and_infreq))


# fetches the most frequent words from the documents
def get_top_n_words(documents, nbr_of_top_words=None):
    vec = CountVectorizer().fit(documents.data)
    bag_of_words = vec.transform(documents.data)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:nbr_of_top_words]


# removes stopwords from newsgroup
def remove_stopwords(newsgroup):
    print("Removal in progress...")
    remove_regex_words(newsgroup)
    i = 0
    with open('../assets/stopwords.txt', 'r') as f:
        words = f.read().split("\n")
        while i < len(words):
            j = 0
            while j < len(newsgroup.data):
                newsgroup.data[j] = re.sub(r'\b' + words[i] + '\s', ' ', newsgroup.data[j])
                j += 1
            i += 1
    f.close()
    print("Stopwords removed")


# remove stopwords containing a ' using regex.
def remove_regex_words(newsgroup):
    i = 0
    with open('../assets/stopwords_regex.txt', 'r') as f:
        words = f.read().split("\n")
        print("Regex in progress...")
        while i < len(words):
            j = 0
            while j < len(newsgroup.data):
                newsgroup.data[j] = re.sub(r'[^\w,\']', " ",  newsgroup.data[j])  # removes special characters except '
                newsgroup.data[j] = re.sub(r'\b' + words[i] + '\s', ' ', newsgroup.data[j])
                newsgroup.data[j] = re.sub(r'[^\w]', " ",  newsgroup.data[j])  # removes special characters
                j += 1
            i += 1
    f.close()
    print("Regex finished")



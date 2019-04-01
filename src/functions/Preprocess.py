import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
from sklearn.datasets import fetch_20newsgroups


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
        remove_stopwords(trainingdata)
        remove_stopwords(testdata)
        lemmatize_newsgroup(trainingdata, testdata, categories[i])
        i += 1


# Determines how a word shall change to convert it to its base form
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Lemmatize and prints newsgroups documents to file
def lemmatize_newsgroup(newsgroups_train, newsgroups_test, category):
    i = 0
    lemmatizer = WordNetLemmatizer()
    size = len(newsgroups_train.data)
    print("Lemmatization in progress...")
    print(category + " training data: ", i, "/", size)
    with open('../assets/20newsgroups/train/newsgroups_train_' + category + '.txt', 'w') as f:
        while i < len(newsgroups_train.data):
            newsgroups_train.data[i] = newsgroups_train.data[i].lower()
            newsgroups_train.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                                  nltk.word_tokenize(newsgroups_train.data[i]) if w not in
                                                  string.punctuation]))
            f.write("%s\n" % newsgroups_train.data[i].encode("utf-8"))
            i += 1
            print(category + " training data: ", i, "/", size)
    with open('../assets/20newsgroups/test/newsgroups_test_' + category + '.txt', 'w') as f:
        size = len(newsgroups_test.data)
        i = 0
        print(category + " test data: ", i, "/", size)
        while i < len(newsgroups_test.data):
            newsgroups_test.data[i] = newsgroups_test.data[i].lower()
            newsgroups_test.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                                 nltk.word_tokenize(newsgroups_test.data[i]) if w not in
                                                 string.punctuation]))
            f.write("%s\n" % newsgroups_test.data[i].encode("utf-8"))
            i += 1
            print(category + " test data: ", i, "/", size)
    print("Lemmatization finished")


def print_v2_docs(categories):
    i = 0
    removed = 0
    underline = 0
    while i < len(categories):
        with open('../assets/20newsgroups/test2/newsgroups_test_' + categories[i] + '.txt', 'w') as f:
            lines = [line.rstrip('\n') for line in open('../assets/20newsgroups/test/newsgroups_test_'
                                                        + categories[i] + '.txt')]
            j = 0
            while j < len(lines):
                lines[j] = re.sub(r'[^\w]', " ", lines[j])
                lines[j] = lines[j].strip() + "\n"
                size = len(lines[j])
                lines[j] = lines[j][1:size]
                if len(lines[j]) > 3:
                    f.write(lines[j])
                else:
                    removed += 1
                j += 1
            f.close()
        with open('../assets/20newsgroups/train2/newsgroups_train_' + categories[i] + '.txt', 'w') as f:
            lines = [line.rstrip('\n') for line in open('../assets/20newsgroups/train/newsgroups_train_'
                                                        + categories[i] + '.txt')]
            j = 0
            while j < len(lines):
                lines[j] = re.sub(r'[^\w]', " ", lines[j])
                lines[j] = lines[j].strip() + "\n"
                size = len(lines[j])
                lines[j] = lines[j][1:size]
                if len(lines[j]) > 3:
                    f.write(lines[j])
                else:
                    removed += 1
                j += 1
            f.close()
        i += 1
    print("Removed:", removed)
    print("Underline:", underline)


# get stopwords from file
def get_stopwords():
    f = open('./assets/stopwords.txt')  # https://github.com/suzanv/termprofiling/blob/master/stoplist.txt
    x = f.read().split("\n")
    f.close()
    return x


# removed stopwords from newsgroup
def remove_stopwords(newsgroup):
    print("Removal in progress...")
    remove_regex_words(newsgroup)
    i = 0
    with open('../assets/stopwords.txt', 'r') as f:
        words = f.read().split("\n")
        while i < len(words):
            j = 0
            while j < len(newsgroup.data):
                newsgroup.data[j] = re.sub(words[i], '', newsgroup.data[j])
                j += 1
            i += 1
    f.close()
    print("Stopwords removed")


# remove stopwords with a ' using regex.
def remove_regex_words(newsgroup):
    i = 0
    with open('../assets/stopwords_regex.txt', 'r') as f:
        words = f.read().split("\n")
        print("Regex in progress...")
        while i < len(words):
            j = 0
            while j < len(newsgroup.data):
                newsgroup.data[j] = re.sub(words[i], '', newsgroup.data[j])
                j += 1
            i += 1
    f.close()
    print("Regex finished")



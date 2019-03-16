import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string


# Preprocess the 20 Newsgroups data
def process(trainingdata, testdata):
    remove_regex_words(trainingdata)
    lemmatize_newsgroup(trainingdata, testdata)


# Determines how a word shall change to convert it to its base form
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Lemmatize newsgroups documents
def lemmatize_newsgroup(newsgroups_train, newsgroups_test):
    i = 0
    lemmatizer = WordNetLemmatizer()
    size = len(newsgroups_train.data)
    print("Lemmatization in progress...")
    print("Training data: ", i, "/", size)
    with open('../assets/newsgroups_train.txt', 'w') as f:
        while i < len(newsgroups_train.data):
            newsgroups_train.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                                  nltk.word_tokenize(newsgroups_train.data[i]) if w not in
                                                  string.punctuation]))
            f.write("%s\n" % newsgroups_train.data[i].encode("utf-8"))
            i += 1
            print("Training data: ", i, "/", size)
    i = 0
    with open('../assets/newsgroups_test.txt', 'w') as f:
        size = len(newsgroups_test.data)
        print("Test data: ", i, "/", size)
        while i < len(newsgroups_test.data):
            newsgroups_test.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                                 nltk.word_tokenize(newsgroups_test.data[i]) if w not in
                                                 string.punctuation]))
            f.write("%s\n" % newsgroups_test.data[i].encode("utf-8"))
            i += 1
            print("Test data: ", i, "/", size)
    print("Lemmatization finished")


# loads newsgroups from files
def load_improved_newsgroup(newsgroups_train, newsgroups_test):
    lines = [line.rstrip('\n') for line in open('../assets/newsgroups_train.txt')]
    i = 0
    size = len(lines)
    print("Training data is loading...")
    print("Training data: ", i, "/", size)
    while i < len(lines):
        length = len(lines[i])
        newsgroups_train.data[i] = lines[i][1:length]  # Removes the b
        i += 1
        print("Training data: ", i, "/", size)
    lines = [line.rstrip('\n') for line in open('../assets/newsgroups_test.txt')]
    i = 0
    size = len(lines)
    print("Test data is loading...")
    print("Training data: ", i, "/", size)
    while i < len(lines):
        length = len(lines[i])
        newsgroups_test.data[i] = lines[i][1:length]  # Removes the b
        i += 1
        print("Training data: ", i, "/", size)
    print("All data is loaded")


# get stopwords from file
def get_stopwords():
    f = open('../assets/stopwords.txt')  # https://github.com/suzanv/termprofiling/blob/master/stoplist.txt
    x = f.read().split("\n")
    f.close()
    return x


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



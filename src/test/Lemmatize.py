from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import time

# specific categories for testing (faster load time)
'''
categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']
'''
categories = ['alt.atheism']

# fetch dataset
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers'),
                                     categories=categories)
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_newsgroup():
    i = 0
    time_before = time.time()
    with open('../assets/newsgroups_train.txt', 'w') as f:
        while i < len(newsgroups_train.data):
            newsgroups_train.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                                  nltk.word_tokenize(newsgroups_train.data[i]) if w not in
                                                  string.punctuation]))
            print(i)
            f.write("%s\n" % newsgroups_train.data[i])
            i += 1
    i = 0
    with open('../assets/newsgroups_test.txt', 'w') as f:
        while i < len(newsgroups_test.data):
            newsgroups_test.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                                 nltk.word_tokenize(newsgroups_test.data[i]) if w not in
                                                 string.punctuation]))
            print(i)
            f.write("%s\n" % newsgroups_test.data[i])
            i += 1
    time_after = time.time()
    print("Time:", time_after-time_before, " seconds")


def load_improved_newsgroup():
    lines = [line.rstrip('\n') for line in open('../assets/newsgroups_train.txt')]
    i = 0
    while i < len(lines):
        newsgroups_train.data[i] = lines[i]
        i += 1
    lines = [line.rstrip('\n') for line in open('../assets/newsgroups_test.txt')]
    i = 0
    while i < len(lines):
        newsgroups_test.data[i] = lines[i]
        i += 1


text = ([newsgroups_train.data[0]])
lemmatize_newsgroup()
print(text)
print(newsgroups_train.data[0])
newsgroups_train.data[0] = ''
print(newsgroups_train.data[0])
load_improved_newsgroup()
print(newsgroups_train.data[0])

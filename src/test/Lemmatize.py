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


# Determines how a word shall change to convert it to its base form
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Lemmatize and prints newsgroups documents to file
def lemmatize_newsgroup(train, test, category):
    i = 0
    lemmatizer = WordNetLemmatizer()
    size = len(newsgroups_train.data)
    print("Lemmatization in progress...")
    print(category + " training data: ", i, "/", size)
    while i < len(train.data):
        train.data[i] = train.data[i].lower()
        train.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                   nltk.word_tokenize(train.data[i]) if w not in string.punctuation]))
        i += 1
        print(category + " training data: ", i, "/", size)
    size = len(test.data)
    i = 0
    print(category + " test data: ", i, "/", size)
    while i < len(test.data):
        test.data[i] = test.data[i].lower()
        test.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                  nltk.word_tokenize(test.data[i]) if w not in string.punctuation]))
        i += 1
        print(category + " test data: ", i, "/", size)
    print("Lemmatization finished")


text = ([newsgroups_train.data[0]])
newsgroups_train.data[0] = "were, be, is, are"
lemmatize_newsgroup(newsgroups_train, newsgroups_test, categories[0])
print(text)
print(newsgroups_train.data[0])

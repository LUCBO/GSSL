from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import time

# specific categories for testing (faster load time)
categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

# fetch dataset
newsgroups_train = fetch_20newsgroups(subset='train',
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


def lemmatize_newsgroup(newsgroups):
    i = 0
    time_before = time.time()
    while i < len(newsgroups.data):
        newsgroups.data[i] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                        nltk.word_tokenize(newsgroups.data[i]) if w not in string.punctuation]))
        print(i)
        i += 1
    time_after = time.time()
    print("Time:", time_after-time_before, " seconds")


text = ([newsgroups_train.data[0]])
text2 = ([newsgroups_train.data[2033]])
lemmatize_newsgroup(newsgroups_train)
print(text)
print(newsgroups_train.data[0])
print(text2)
print(newsgroups_train.data[2033])

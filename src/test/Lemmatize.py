from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string

# specific categories for testing (faster load time)
categories = ['alt.atheism', 'talk.religion.misc',
              'comp.graphics', 'sci.space']

# fetch dataset
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers'),
                                      categories=categories)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


lemmatizer = WordNetLemmatizer()
print([newsgroups_train.data[0]])
word_list = nltk.word_tokenize(newsgroups_train.data[0])
newsgroups_train.data[0] = (" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in
                                      nltk.word_tokenize(newsgroups_train.data[0]) if w not in string.punctuation]))
print(newsgroups_train.data[0])

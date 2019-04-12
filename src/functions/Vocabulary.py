from sklearn.feature_extraction.text import CountVectorizer
from src.classes.dataset import Dataset


def get_vocabulary(categories):
    i = 0
    voc_list = []
    while i < len(categories):
        lines = [line.rstrip('\n') for line in open('../assets/vocabulary/vocabulary_'
                                                    + categories[i] + '.txt')]
        j = 0
        while j < len(lines):
            voc_list.extend([lines[j]])
            j += 1
        i += 1
    voc_list = list(dict.fromkeys(voc_list))  # remove duplicates
    return voc_list


def get_vocabulary_only_labeled(categories):
    i = 0
    voc_list = []
    while i < len(categories):
        lines = [line.rstrip('\n') for line in open('../assets/vocabulary_labeled/vocabulary_'
                                                    + categories[i] + '.txt')]
        j = 0
        while j < len(lines):
            voc_list.extend([lines[j]])
            j += 1
        i += 1
    voc_list = list(dict.fromkeys(voc_list))  # remove duplicates
    return voc_list


def create_vocabulary(categories, size):
    i = 0
    while i < len(categories):
        dataset = Dataset([categories[i]])
        freq_words = get_top_n_words(dataset.train['data'], size)
        with open('../assets/vocabulary/vocabulary_' + categories[i] + '.txt', 'w') as f:
            j = 0
            while j < len(freq_words):
                f.write(freq_words[j][0] + '\n')
                j += 1
            f.close()
        i += 1


def create_vocabulary_only_labeled(dataset, category, size):
    freq_words = get_top_n_words(dataset.train['data'], size)
    with open('../assets/vocabulary_labeled/vocabulary_' + category + '.txt', 'w') as f:
        j = 0
        while j < len(freq_words):
            f.write(freq_words[j][0] + '\n')
            j += 1
        f.close()


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]



from sklearn.feature_extraction.text import CountVectorizer
from src.classes.dataset import Dataset


# fetches the vocabulary for all the training documents
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


# fetches the vocabulary for only the labeled training documents
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


# creates a vocabulary using all the training documents
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


# creates a vocabulary using only the labeled training documents
# made during runtime
def create_vocabulary_only_labeled(dataset, category, size):
    freq_words = get_top_n_words(dataset.train['data'], size)
    with open('../assets/vocabulary_labeled/vocabulary_' + category + '.txt', 'w') as f:
        j = 0
        while j < len(freq_words):
            f.write(freq_words[j][0] + '\n')
            j += 1
        f.close()


# fetches the most frequent words from the documents
def get_top_n_words(documents, nbr_of_top_words=None):
    vec = CountVectorizer().fit(documents)
    bag_of_words = vec.transform(documents)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:nbr_of_top_words]



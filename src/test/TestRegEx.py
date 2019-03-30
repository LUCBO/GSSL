import re
from sklearn.datasets import fetch_20newsgroups


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


trainingdata = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes'),
                                  categories=['alt.atheism'])
trainingdata.data[0] = "I'm don't like that I can't do what you'll do"
before = trainingdata.data[0]
remove_regex_words(trainingdata)
print([before])
print([trainingdata.data[0]])


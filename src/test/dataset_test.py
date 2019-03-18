from src.classes.dataset import Dataset
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'talk.religion.misc']
dataset = Dataset(categories)
dataset.split_train(5, shuffle=False)

print('----------SPLIT WITHOUT SHUFFLE-------------')
for i in range(len(dataset.train['data'])):
    print('Text: ', dataset.train['data'][i][:10], ' Target: ', dataset.train['target'][i])

print('----------WITH SHUFFLE-------------')
dataset.train = dataset.shuffle(dataset.train)
for i in range(len(dataset.train['data'])):
    print('Text: ', dataset.train['data'][i][:10], ' Target: ', dataset.train['target'][i])

print('----------LABELED/UNLABELED-------------')
labeled = 0
for i in range(len(dataset.train['data'])):
    if dataset.train['target'][i] != -1:
        labeled += 1
        print(dataset.train['target'][i])
print('Labeled: ', labeled)

unlabeled = 0
for i in range(len(dataset.train['data'])):
    if dataset.train['target'][i] == -1:
        unlabeled += 1
print('Unlabeled: ', unlabeled)

train_20newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
print('Total in original dataset: ', len(train_20newsgroups.target))

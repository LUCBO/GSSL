from src.classes.dataset import Dataset

dataset = Dataset(['alt.atheism', 'talk.religion.misc'])
dataset.split(5, shuffle=False)

print('----------SPLIT WITHOUT SHUFFLE-------------')
for i in range(len(dataset.train['data'])):
    print('Text: ', dataset.train['data'][i][:10], ' Target: ', dataset.train['target'][i])

print('----------WITH SHUFFLE-------------')
dataset.train = dataset.shuffle(dataset.train)
for i in range(len(dataset.train['data'])):
    print('Text: ', dataset.train['data'][i][:10], ' Target: ', dataset.train['target'][i])
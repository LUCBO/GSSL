from sklearn.datasets import fetch_20newsgroups
import random


class Dataset:
    def __init__(self, categories=None):
        self.categories = fetch_20newsgroups(subset='train', categories=categories).target_names  # List of categories in dataset

        self.train = {  # Labeled dataset
            'data': [],
            'target': []
        }

        self.test = {  # "Unlabeled" dataset
            'data': [],
            'target': []
        }

    def split(self, category_size, shuffle=True):
        """
        Split training dataset into train and test datasets
        :param category_size: Integer. Number of documents per category
        :param shuffle: Boolean. True will shuffle train and test dataset after split
        :return:
        """
        target_label = 0
        for category in self.categories:
            fetch_dataset = fetch_20newsgroups(subset='train',
                                               remove=('headers', 'footers', 'quotes'),
                                               categories=[category])

            # Pick random documents from category dataset into new train dataset
            fetch_dataset_length = len(fetch_dataset.data)
            for i in range(category_size):
                indexes_left = (fetch_dataset_length - i) - 1
                if indexes_left >= 0:
                    random_index = random.randint(0, indexes_left)
                    data = fetch_dataset.data[random_index]

                    self.train['data'].append(data)
                    self.train['target'].append(target_label)

                    del fetch_dataset.data[random_index]
                else:
                    break

            # Rest of category dataset goes into new test dataset
            self.test['data'].extend(fetch_dataset.data)
            self.test['target'] += [target_label] * len(fetch_dataset.data)
            target_label += 1

        if shuffle:
            self.train = self.shuffle(self.train)
            self.test = self.shuffle(self.test)

    def shuffle(self, dataset):
        """
        Shuffle both data and targets
        :param dataset: Dataset that should be shuffled
        :return: Shuffled dataset
        """
        merge_X_Y = list(zip(dataset['data'], dataset['target']))
        random.shuffle(merge_X_Y)
        x, y = zip(*merge_X_Y)

        return {
            'data': x,
            'target': y
        }

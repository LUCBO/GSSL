from sklearn.datasets import fetch_20newsgroups
import random


class Dataset:
    def __init__(self, categories=None, preprocessed=True):
        self.train = {
            'data': [],
            'target': [],
            'target_names': []
        }

        self.test = {
            'data': [],
            'target': [],
            'target_names': []
        }

        if preprocessed:
            self.load_preprocessed(categories)
        else:
            self.load_original(categories)

    def split_train(self, category_size, shuffle=True):
        """
        Split training dataset into labeled and unlabeled
        :param category_size: Integer. Number of labeled documents per category
        :param shuffle: Boolean. True will shuffle dataset after split
        :return:
        """
        dataset = {
            'data': [],
            'target': [],
            'target_names': self.train['target_names']
        }
        target_label = 0
        for category in self.train['target_names']:
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

                    dataset['data'].append(data)
                    dataset['target'].append(target_label)

                    del fetch_dataset.data[random_index]
                else:
                    break

            # Rest of category dataset goes into new test dataset
            dataset['data'].extend(fetch_dataset.data)
            dataset['target'] += [-1] * len(fetch_dataset.data)  # Set -1 (unlabeled) on rest
            target_label += 1

        if shuffle:
            self.train = self.shuffle(dataset)
        else:
            self.train = dataset

    def split_train_true(self, category_size, shuffle=True):
        """
        Split training dataset into labeled and unlabeled
        :param category_size: Integer. Number of labeled documents per category
        :param shuffle: Boolean. True will shuffle dataset after split
        :return:
        """
        dataset = {
            'data': [],
            'target': [],
            'target_names': self.train['target_names']
        }
        target_label = 0
        for category in self.train['target_names']:
            fetch_dataset = self.load_train_for_split(category, target_label)

            # Pick random documents from category dataset into new train dataset
            fetch_dataset_length = len(fetch_dataset)
            for i in range(category_size):
                indexes_left = (fetch_dataset_length - i) - 1
                if indexes_left >= 0:
                    random_index = random.randint(0, indexes_left)
                    data = fetch_dataset[random_index][0]

                    dataset['data'].append(data)
                    dataset['target'].append(target_label)

                    del fetch_dataset[random_index]
                else:
                    break

            # Rest of category dataset goes into new test dataset
            i = 0
            while i < len(fetch_dataset):
                dataset['data'].append(fetch_dataset[i][0])
                dataset['target'] += [-1]  # Set -1 (unlabeled) on rest
                i += 1

            target_label += 1

        if shuffle:
            self.train = self.shuffle(dataset)
        else:
            self.train = dataset

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
            'target': y,
            'target_names': dataset['target_names']
        }

    def load_preprocessed(self, categories):
        print('Loading preprocessed dataset..')

        # Load training dataset
        for i, category in enumerate(categories):
            file = open('./assets/20newsgroups/train2/newsgroups_train_' + category + '.txt')
            lines = [line.rstrip('\n') for line in file]

            self.train['data'].extend(lines)
            self.train['target'] += [i] * len(lines)
            self.train['target_names'].append(category)
            file.close()

        # Load testing dataset
        for i, category in enumerate(categories):
            file = open('./assets/20newsgroups/test2/newsgroups_test_' + category + '.txt')
            lines = [line.rstrip('\n') for line in file]

            self.test['data'].extend(lines)
            self.test['target'] += [i] * len(lines)
            self.test['target_names'].append(category)
            file.close()

        print('Load completed!')

    def load_train_for_split(self, category, target):
        file = open('./assets/20newsgroups/train2/newsgroups_train_' + category + '.txt')
        lines = [line.rstrip('\n') for line in file]
        newsgroup_train = []
        i = 0

        while i < len(lines):
            newsgroup_train.append([lines[i], target, category])
            i += 1
        file.close()
        return newsgroup_train

    def load_original(self, categories):
        print('Loading original dataset..')

        train_20newsgroups = fetch_20newsgroups(subset='train',
                                                remove=('headers', 'footers', 'quotes'),
                                                categories=categories)
        self.train = {
            'data': train_20newsgroups.data,
            'target': train_20newsgroups.target,
            'target_names': train_20newsgroups.target_names
        }

        train_20newsgroups = fetch_20newsgroups(subset='test',
                                                remove=('headers', 'footers', 'quotes'),
                                                categories=categories)
        self.test = {
            'data': train_20newsgroups.data,
            'target': train_20newsgroups.target,
            'target_names': train_20newsgroups.target_names
        }

        print('Load completed!')


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


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

    def split(self, category_size):
        """
        Split training dataset into train and test for semi-supervised learning
        :param category_size: Number of random documents per category in training dataset
        :return:
        """

        for category in self.categories:
            fetch_dataset = fetch_20newsgroups(subset='train',
                                               remove=('headers', 'footers', 'quotes'),
                                               categories=[category])

            xTrain, xTest, yTrain, yTest = train_test_split(fetch_dataset.data, fetch_dataset.target,
                                                            train_size=category_size, random_state=0)

            self.train['data'].extend(xTrain)
            self.train['target'].extend(yTrain)
            self.test['data'].extend(xTest)
            self.test['target'].extend(yTest)



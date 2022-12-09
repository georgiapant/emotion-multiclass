import pandas as pd
from collections import Counter
import json
from sklearn.model_selection import train_test_split
import numpy as np
from src.features.translate import oversample_with_back_translation


class GetData:
    def __init__(self, project_root_path, RANDOM_SEED):
        self.project_root_path = project_root_path
        self.RANDOM_SEED = RANDOM_SEED

    def ekman(self, split_in=3, oversampling=False):
        # Load data and set labels
        data = pd.read_csv(self.project_root_path + "/data/external/ekman_emotions.csv", encoding="utf8",
                           low_memory=False)

        data = data[['sentiment', 'content']]
        print('Dataset shape %s' % Counter(data['sentiment']), flush=True)

        # Create statistics of dataset
        stats = pd.DataFrame()
        stats['count'] = data.groupby('sentiment').count()['content']
        stats['percent'] = 100 * stats['count'] / len(data)
        stats['sentiment'] = stats.index
        stats = stats[['count', 'percent', 'sentiment']]
        # stats.plot.pie(y='percent')
        stats = stats.reset_index(drop=True)
        print(stats)

        # Transform text labels to numbers
        # d = dict(zip(emotions, range(0, num_labels)))

        emotions = np.unique(data['sentiment'])
        with open(self.project_root_path + "/data/resources/emotion_mappings.json") as file:
            file_data = file.read()
            mapping = json.loads(file_data)

        # mapping = json.loads(r"../../data/interim/emotion_mappings.json")
        data['label'] = data['sentiment'].map(mapping, na_action='ignore').astype('int64')
        data.drop(['sentiment'], inplace=True, axis=1)
        print(mapping)

        # weights = calculating_class_weights(data.drop(columns=['content']), np.unique(data['label']))

        if oversampling:

            train, test = train_test_split(data, test_size=0.2, random_state=self.RANDOM_SEED, stratify=data['label'].values)
            val, test = train_test_split(test, test_size=0.5, random_state=self.RANDOM_SEED, stratify=test['label'].values)

            oversampled_train = oversample_with_back_translation(train)
            X_train = oversampled_train['content'].values
            y_train = oversampled_train['label'].values.tolist()
            X_val = val['content'].values
            y_val = val['label'].values.tolist()
            X_test = test['content'].values
            y_test = test['label'].values.tolist()
            return X_train, y_train, X_val, y_val, X_test, y_test, emotions

        else:

            X = data['content'].values
            y = data['label'].values

            # load train, test and validation data]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.RANDOM_SEED,
                                                                stratify=y)
            if split_in == 3:
                X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=self.RANDOM_SEED,
                                                            stratify=y_test)
                return X_train, y_train, X_val, y_val, X_test, y_test, emotions  # , weights
            elif split_in == 2:
                return X_train,y_train ,X_test, y_test, emotions
            else:
                return "No possible split"

    def merged(self, split_in=3):

        # Load data and set labels
        data = pd.read_csv(
            self.project_root_path + "data/external/emotions_merged.csv",
            encoding="utf8", low_memory=False)

        data = data[['sentiment', 'content']]

        print('Original dataset shape %s' % Counter(data['sentiment']), flush=True)

        # Create statistics of dataset
        stats = pd.DataFrame()
        stats['count'] = data.groupby('sentiment').count()['content']
        stats['percent'] = 100 * stats['count'] / len(data)
        stats['sentiment'] = stats.index
        stats = stats[['count', 'percent', 'sentiment']]
        stats = stats.reset_index(drop=True)
        print(stats)

        # Transform text labels to numbers
        emotions = np.unique(data['sentiment'])

        d = dict(zip(emotions, range(0, len(emotions))))
        data['label'] = data['sentiment'].map(d, na_action='ignore').astype('int64')
        data.drop(['sentiment'], inplace=True, axis=1)

        X = data['content'].values
        y = data['label'].values

        # load train, test and validation data]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.RANDOM_SEED, stratify=y)

        if split_in == 3:
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=self.RANDOM_SEED,
                                                            stratify=y_test)

            return X_train, y_train, X_val, y_val, X_test, y_test, emotions
        elif split_in == 2:
            return X_train, y_train, X_test, y_test, emotions
        else:
            return "No possible split"


    def isear(self, split_in=3):
        # Load data and set labels

        data = pd.read_csv(
            self.project_root_path + "data/external/ISEAR_DATA.csv",
            encoding="utf8", low_memory=False)

        data = data[['SIT', 'Field1']]
        data.rename(columns={'SIT': 'Text', 'Field1': 'emotion'}, inplace=True)

        X = data['Text'].values
        y = data.drop(columns=['Text'])
        y = y['emotion']

        stats = pd.DataFrame()
        stats['count'] = y.value_counts()
        stats['percent'] = 100 * stats['count'] / len(y)
        stats['emotion'] = stats.index
        stats = stats[['count', 'percent', 'emotion']]
        # stats.plot.pie(y='percent')
        stats = stats.reset_index(drop=True)
        print(stats)

        # Transform text labels to numbers
        emotions = np.unique(y)
        d = dict(zip(emotions, range(0, len(emotions))))
        print(d)
        y['label'] = y.map(d, na_action='ignore').astype('int64')
        y = y['label'].values

        # load train, test and validation data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=self.RANDOM_SEED, stratify=y)

        if split_in == 3:
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=self.RANDOM_SEED,
                                                            stratify=y_test)

            return X_train, y_train, X_val, y_val, X_test, y_test, emotions

        elif split_in == 2:
            return X_train, y_train, X_test, y_test, emotions
        else:
            return "No possible split"
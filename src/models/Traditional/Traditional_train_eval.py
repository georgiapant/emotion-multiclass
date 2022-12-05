from src.data.create_dataset import GetData
from src.features.preprocess_feature_creation import tf_idf
from src.evaluate import evaluate
from src.models.Traditional.Models import lsvc, lr, rf, nb
import pickle
from sklearn.pipeline import Pipeline

class ML:

    def __init__(self, dataset, BATCH_SIZE, MAX_LEN, EPOCHS, patience, BERT_MODEL, RANDOM_SEED,
                 project_root_path, es, word_embd_type, analyzer, ngram_range):

        self.RANDOM_SEED = RANDOM_SEED
        self.project_root_path = project_root_path
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        create_dataset = GetData(self.project_root_path, self.RANDOM_SEED)

        if dataset == 'ekman':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels \
                = create_dataset.ekman()
        elif dataset == 'isear':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels \
                = create_dataset.isear()
        elif dataset == 'merged':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.labels \
                = create_dataset.merged()
        else:
            print("No dataset with the name {}".format(dataset))

        self.num_labels = len(self.labels)

    def main(self, model_name):
        X_train_vect, X_test_vect, vect = tf_idf(self.X_train, self.X_test, self.analyzer, self.ngram_range)

        if model_name == "NB":
            model = nb(X_train_vect, self.y_train)
        elif model_name == 'LR':
            model = lr(X_train_vect, self.y_train)
        elif model_name == 'RF':
            model = rf(X_train_vect, self.y_train)
        elif model_name == 'SVC':
            model = lsvc(X_train_vect, self.y_train)
        else:
            print("No Known model")
            model = None

        predictions = model.predict(X_test_vect)
        evaluate(predictions, self.y_test)

        # Save model
        save_model = Pipeline([('tfidf', vect), ('clf', model)])
        filename = self.project_root_path + "/models/" + model_name + ".sav"
        pickle.dump(save_model, open(filename, 'wb'))
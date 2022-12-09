import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from src.features.preprocess_feature_creation import text_preprocessing, create_embedding_matrix
from src.evaluate import evaluate
from src.models.CNN.CNN_model import cnn
from src.models.LSTM.LSTM_model import lstm
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def predict(X_test, y_test, labels, model_name, project_root_path, MAX_LEN):
    # load tokenizer
    with open(project_root_path + '/models/' + model_name + '_tokenizer', 'rb') as handle:
        tok = pickle.load(handle)

    # load pretrained model
    model = load_model(project_root_path + '/models/' + model_name + '.h5', compile=False)

    # preprocessing
    X_test = [' '.join(text_preprocessing(text)) for text in X_test]

    # tokenization
    sequences = tok.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(sequences, maxlen=MAX_LEN)

    # prediction
    probabilities = model.predict(np.array(X_test_pad))

    evaluate(probabilities, y_test, labels=labels)


def train(X_train, y_train, X_dev, y_dev, model_name, es, patience, EPOCHS, BATCH_SIZE, embed_type, EMBED_NUM_DIMS, MAX_SEQ_LEN,
          num_labels, project_root_path):
    X_train_pad, X_dev_pad, vocab_size, embedding_matrix = create_embedding_matrix(X_train, X_dev, model_name,
                                                                                   embed_type=embed_type,
                                                                                   MAX_SEQ_LEN=MAX_SEQ_LEN,
                                                                                   EMBED_NUM_DIMS=EMBED_NUM_DIMS,
                                                                                   project_root_path=project_root_path)

    y_train_categorical = to_categorical(y_train)
    y_dev_categorical = to_categorical(y_dev)

    try:
        if model_name == 'CNN':
            model = cnn(vocab_size, embedding_matrix, EMBED_NUM_DIMS, MAX_SEQ_LEN, num_labels)
        elif model_name == 'LSTM':
            model = lstm(vocab_size, embedding_matrix, EMBED_NUM_DIMS, MAX_SEQ_LEN, num_labels)
        else:
            print("No known model")
            model = None

        es = EarlyStopping(monitor=es, mode='min', verbose=1, patience=patience)
        # save the best model
        mc = ModelCheckpoint(project_root_path + 'models/' + model_name + '.h5', monitor='val_loss', mode='min', verbose=1,
                             save_best_only=True)
        model.fit(np.array(X_train_pad), np.array(y_train_categorical), epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(np.array(X_dev_pad), np.array(y_dev_categorical)), callbacks=[es, mc])

    except Exception as e:
        print(e)
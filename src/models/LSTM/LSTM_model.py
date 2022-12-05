from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout


def lstm(vocab_size, embedding_matrix, EMBED_NUM_DIMS, MAX_SEQ_LEN, num_labels):
    bidirectional = True

    embedding_layer = Embedding(vocab_size, EMBED_NUM_DIMS, input_length=MAX_SEQ_LEN, weights=[embedding_matrix],
                                trainable=False)

    # Embedding Layer, LSTM or biLSTM, Dense, softmax
    model = Sequential()
    model.add(embedding_layer)

    if bidirectional:
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
    else:
        model.add((LSTM(64, return_sequences=False)))

    model.add(Dropout(0.2))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model
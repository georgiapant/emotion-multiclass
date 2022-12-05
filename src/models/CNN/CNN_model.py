from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Flatten, Dropout


def cnn(vocab_size, embedding_matrix, EMBED_NUM_DIMS, MAX_SEQ_LEN, num_labels):
    embedding_layer = Embedding(vocab_size, EMBED_NUM_DIMS, input_length=MAX_SEQ_LEN, weights=[embedding_matrix],
                                trainable=False)

    # Convolution
    kernel_size = 3
    filters = 256
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
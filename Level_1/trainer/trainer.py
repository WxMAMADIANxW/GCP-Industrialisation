import datetime
import logging
import tempfile
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

sentiment_mapping = {
    0: "negative",
    2: "neutral",
    4: "positive"
}

CLASSES = {'negative': 0, 'positive': 1}  # label-to-int mapping
VOCAB_SIZE = 25000  # Limit on the number vocabulary size used for tokenization
MAX_SEQUENCE_LENGTH = 50  # Sentences will be truncated/padded to this length


def read_preprocess_data(uri):
    """Read input data and embedded .
    Args:
      uri(String): GCP Path to input and embedding_matrix.
    Returns:
      x_train, y_train, embedding_matrix(Pandas Data Frame): Training , test and embedded dataset.
    """
    x_train = pd.read_csv(os.path.join(uri, 'x_train.csv'), encoding="latin1", header=None)
    y_train = pd.read_csv(os.path.join(uri, 'y_train.csv'), encoding="latin1", header=None).to_numpy()
    embedding_matrix = pd.read_csv(os.path.join(uri, 'embedding_matrix.csv'), encoding="latin1", header=None).to_numpy()

    return x_train, y_train, embedding_matrix


def split_input(sents, labels, test_size=0.1):
    """
        Split Data to Train and test dataset
    """
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(sents, labels, test_size=test_size)

    # Create vocabulary from training corpus.

    return y_train, y_test, X_train, X_test


def create_model(vocab_size, embedding_dim, filters, kernel_sizes, dropout_rate, pool_size, embedding_matrix):
    """
       Create TensorFlow Keras CNN model
   """
    # Input layer
    model_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # Embedding layer
    z = tf.keras.layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=embedding_dim,
        input_length=MAX_SEQUENCE_LENGTH,
        weights=[embedding_matrix]
    )(model_input)

    z = tf.keras.layers.Dropout(dropout_rate)(z)

    # Convolutional block
    conv_blocks = []
    for kernel_size in kernel_sizes:
        conv = tf.keras.layers.Convolution1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            activation="relu",
            bias_initializer='random_uniform',
            strides=1)(z)
        conv = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
        conv = tf.keras.layers.Flatten()(conv)
        conv_blocks.append(conv)

    z = tf.keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(100, activation="relu")(z)
    model_output = tf.keras.layers.Dense(1, activation="sigmoid")(z)

    model = tf.keras.models.Model(model_input, model_output)

    return model


def train_evaluate_explain_model(hparams):
    """Train, evaluate, explain TensorFlow Keras DNN Regressor.
    Args:
      hparams(dict): A dictionary containing model training arguments.
    Returns:
      history(tf.keras.callbacks.History): Keras callback that records training event history.
    """

    EMBEDDING_DIM = 25
    POOL_SIZE = 3
    KERNEL_SIZES = [2, 5, 8]

    x_train_all, y_train_all, embedding_matrix = read_preprocess_data(hparams['preprocess-data-dir'])
    y_train, y_validation, train_vectorized, validation_vectorized = split_input(x_train_all, y_train_all)

    model = create_model(VOCAB_SIZE, EMBEDDING_DIM, hparams['filters'], KERNEL_SIZES, hparams['dropout'], POOL_SIZE,
                         embedding_matrix)
    logging.info(model.summary())
    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Nadam(lr=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    # keras train
    history = model.fit(
        train_vectorized,
        y_train,
        epochs=hparams['n-checkpoints'],
        batch_size=hparams['batch-size'],
        validation_data=(validation_vectorized, y_validation),
        verbose=2,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc',
                min_delta=0.005,
                patience=3,
                factor=0.5),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.005,
                patience=5,
                verbose=0,
                mode='auto'
            ),
            tf.keras.callbacks.History()
        ]
    )

    # Create a temp directory to save intermediate TF SavedModel prior to Explainable metadata creation.
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # Export Keras model in TensorFlow SavedModel format.
    save_model_dir = hparams['model-dir'] + '/candidate_model_' + nowTime
    tf.saved_model.save(model, save_model_dir)

    Path(hparams['output-model-dir']).parent.mkdir(parents=True, exist_ok=True)
    with open(hparams['output-model-dir'], "w") as output_file:
        output_file.write(save_model_dir)
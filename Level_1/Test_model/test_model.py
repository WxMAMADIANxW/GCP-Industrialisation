import tensorflow as tf
import pandas as pd
import datetime
import numpy as np
from pathlib import Path
import os
import logging
import json

logging.getLogger().setLevel('DEBUG')

def read_test_data(uri):
    """Read test data from GCP URI
    Args:
      uri(String): GCP URI path to test data .
    Returns:
      y_test(pandas Dataframe): Test feature
      x_test(pandas Dataframe): Test prediction variable
    """
    logging.debug(f'Downloading test data from {uri}')
    y_test = pd.read_csv(os.path.join(uri + 'y_test.csv'), encoding="latin1", header=None).to_numpy()
    x_test = pd.read_csv(os.path.join(uri + 'x_test.csv'), encoding="latin1", header=None).to_numpy()

    return y_test, x_test


def valid_model(model, y_test, x_test, threshold , uri_save_model):
    """Test  Model .
    Args:
      model(tf.model): tensorflow model.
      y_test:Test feature
      x_test: Test prediction variable
      threshold: threshold for test metric
      uri_save_model: Uri path to save model if verify threshold
    Returns:
         Test metric value
    """
    logging.debug(f'Validating model saved in {uri_save_model}')
    [loss, acc] = model.evaluate(x_test, y_test)
    scores = model.predict(x_test)
    predictions = np.array([int(np.round(i)) for i in scores])
    confu_matrix = tf.math.confusion_matrix(predictions, y_test)
    std_pred = np.std(scores)
    threshold = json.loads(threshold)
    if acc > threshold["acc"] and std_pred > threshold["std"]:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        tf.saved_model.save(model, (uri_save_model) + '_' + nowTime)
    return loss, acc, confu_matrix, std_pred


def evaluate_model(hparams):
    """Train, evaluate, explain TensorFlow Keras DNN Regressor.
    Args:
      hparams(dict): A dictionary containing model training arguments.
    Returns:
      history(tf.keras.callbacks.History): Keras callback that records training event history.
    """

    logging.debug(f'Evaluate model.')
    logging.debug(f'Parameters: {hparams}')

    y_test, x_test = read_test_data(hparams['preprocess-data-dir'])
    model = tf.keras.models.load_model(hparams['model-dir'], custom_objects={'tf': tf})

    [loss, acc, confusion_matrix, std_prediction] = valid_model(model, y_test, x_test, hparams['performance-threshold'], hparams['model-validation-dir'])

    logging.debug(f"Save model performances in {hparams['output-performance-model']}")
    Path(hparams['output-performance-model']).parent.mkdir(parents=True, exist_ok=True)
    with open(hparams['output-performance-model'], "w") as output_file:
        output_file.write(str("loss : " + str(loss) +
                              "/n acc :" + str(acc) +
                              "/n matrix-co :" + str(confusion_matrix) +
                              "/n std-prediction :" + str(std_prediction)))

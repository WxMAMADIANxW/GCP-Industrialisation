import os
import pandas as pd
import numpy as np
from preprocess import TextPreprocessor
import argparse
import logging
import json
import pickle
from sklearn.model_selection import train_test_split
import datetime

from google.cloud.storage import Client

CLASSES = {'negative': 0, 'positive': 1}  # label-to-int mapping
VOCAB_SIZE = 25000  # Limit on the number vocabulary size used for tokenization
MAX_SEQUENCE_LENGTH = 50  # Sentences will be truncated/padded to this length
sentiment_mapping = {
    0: "negative",
    2: "neutral",
    4: "positive"
}


def read_data_uri(uri, start_date, end_date):
    """
        read Tweet data from Gcp URI
         Args:
            uri(String ): Uri to sentimental Data
            Start_date(Sting): Start date of tweet to read
            end_date(String) : End date of tweet to read
        Returns:
            data_input (Padnas Dataframe): Input Data .
    """
    data_input = pd.read_csv(uri, encoding="latin1", header=None) \
        .rename(columns={
        0: "sentiment",
        1: "id",
        2: "time",
        3: "query",
        4: "username",
        5: "text"
    })[["sentiment", "text"]]
    if end_date != '' and start_date != '':
        data_input['time'] = data_input['time'].apply(
            lambda x: datetime.datetime.strptime(x, "%a %b %d %H:%M:%S PDT %Y"))
        data_input = data_input[(data_input['time'].dt.strftime('%Y-%m-%d') >= start_date)
                                & (data_input['time'].dt.strftime('%Y-%m-%d') <= end_date)]
    data_input["sentiment_label"] = data_input["sentiment"].map(sentiment_mapping)
    return data_input


def read_embbeded_data_uri(bucket, uri_data, temp, processor, EMBEDDING_DIM):
    """
       read_embedded_data_uri will read embedded data from Gcp URI
        Args:
           bucket(String ): GCP bucket
           uri_data(Sting): Uri to Embedded Data
           temp(String) : temporal repo
       Returns:
           embedding_matrix: embedded Data .
   """
    client = Client()
    bucket = client.get_bucket(bucket)
    temp_folder = temp
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    blob = bucket.get_blob(uri_data)
    downloaded_file = blob.download_to_filename(temp_folder + '/glove.twitter.27B.50d.txt')

    def get_coaefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coaefs(*o.strip().split()) for o in
                            open(temp_folder + "/glove.twitter.27B.50d.txt", "r", encoding="utf8"))
    word_index = processor._tokenizer.word_index
    nb_words = min(VOCAB_SIZE, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= VOCAB_SIZE: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocess_input(input, bucket, uri_data):
    """
       preprocess_input will clean and tokenize input data
        Args:
           bucket(String ): GCP bucket
           input(Sting): Input data to clean
           uri_data(String) : path where to save data
       Returns:
           processor: preprocessor model .
           train_texts_vectorized : Vectorized data feature
           labels: cleaned Sentimental data
   """
    sents = input.text
    labels = np.array(input.sentiment_label.map(CLASSES))

    # Train and test split

    processor = TextPreprocessor(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
    processor.fit(sents)
    # Preprocess the data
    train_texts_vectorized = processor.transform(sents)

    client = Client()
    bucket = client.get_bucket(bucket)
    logging.debug('Dumping processor pickle')
    blob = bucket.blob(uri_data)
    processor_dump = pickle.dumps(processor)
    blob.upload_from_string(processor_dump)

    return processor, train_texts_vectorized, labels


def split_input(sents, labels, test_size=0.2):
    """
        Split Data to Train and test dataset
    """
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(sents, labels, test_size=test_size)

    # Create vocabulary from training corpus.

    return y_train, y_test, X_train, X_test


def run(hparams):
    logging.debug('preprocessing data Start')
    EMBEDDING_DIM = 50
    logging.debug('loading input data')
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    input_data = read_data_uri(hparams['input-data-uri'],hparams['input-start-date'],hparams['input-end-date'])
    processor, vectorized_input, label = preprocess_input(input_data, hparams['bucket'], hparams['model-dir'])
    logging.debug('preprocessing input data Done')

    logging.debug('loading embedding data')
    embedding_matrix = read_embbeded_data_uri(hparams['bucket'], hparams['uri_data'], hparams['temp-dir'], processor,
                                              EMBEDDING_DIM)
    logging.debug('loading embedding done')

    y_train, y_test, train_vectorized, x_test = split_input(vectorized_input, label)

    pd.DataFrame(y_train).to_csv(hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/y_train.csv',
                                 index=False, header=False)
    logging.debug('saved preprocessed label data in ' + hparams[
        'preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/y_train.csv')

    pd.DataFrame(y_test).to_csv(hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/y_test.csv',
                                index=False, header=False)
    logging.debug('saved preprocessed label data in ' + hparams[
        'preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/y_test.csv')

    pd.DataFrame(train_vectorized).to_csv(
        hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/x_train.csv', index=False, header=False)
    logging.debug('saved preprocessed label data in ' + hparams[
        'preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/x_train.csv')

    pd.DataFrame(x_test).to_csv(hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/x_test.csv',
                                index=False, header=False)
    logging.debug('saved preprocessed label data in ' + hparams[
        'preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/x_test.csv')

    input_data.to_csv(hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/input.csv', index=False)
    logging.debug('saved preprocessed features data in ' + hparams[
        'preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/input.csv')

    pd.DataFrame(label).to_csv(hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/label.csv',
                               index=False, header=False)
    logging.debug('saved preprocessed label data in ' + hparams[
        'preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/label.csv')

    pd.DataFrame(vectorized_input).to_csv(
        hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/vectorized_input.csv', index=False,
        header=False)
    logging.debug(
        'saved preprocessed and vectorized input data in ' + hparams[
            'preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/vectorized_input.csv')

    pd.DataFrame(embedding_matrix).to_csv(
        hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/embedding_matrix.csv', index=False,
        header=False)
    logging.debug('saved embedding data in ' + hparams[
        'preprocess-data-dir'] + '/preprocess-data-' + nowTime + '/embedding_matrix.csv')

    ouputfile = open("preprocess-data-dir.txt", "w")
    ouputfile.write(str(hparams['preprocess-data-dir'] + '/preprocess-data-' + nowTime))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Vertex custom container training args. These are set by Vertex AI during training but can also be overwritten.
    parser.add_argument('--model-dir', dest='model-dir', type=str, help='Model dir.')

    parser.add_argument('--bucket', dest='bucket',
                        default="'rare-result-248415-tweet-sentiment-analysis'", type=str, help='bucket name.')

    parser.add_argument('--preprocess-data-dir', dest='preprocess-data-dir',
                        default="", type=str, help="dirototory where to save preprocess data ")

    parser.add_argument('--input-data-uri', dest='input-data-uri', type=str,
                        help='Training data GCS or BQ URI set during Vertex AI training.')

    parser.add_argument('--input-start-date', dest='input-start-date', type=str,
                        help='The start date for training tweet.')

    parser.add_argument('--input-end-date', dest='input-end-date', type=str,
                        help='The end date for training tweet.')

    parser.add_argument('--uri_data', dest='validation-data-uri', type=str,
                        help='embedding data GCS or BQ URI set during Vertex AI training.')

    parser.add_argument('--temp-dir', dest='temp-dir', type=str,
                        help='Temp dir set during Vertex AI training.')

    parser.add_argument('--load_json', dest='load_json', default='confg.json',
                        help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()

    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    hparams = args.__dict__

    run(hparams)

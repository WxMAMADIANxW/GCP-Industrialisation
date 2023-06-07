import os
import argparse
import json

import trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Vertex custom container training args. These are set by Vertex AI during training but can also be overwritten.
    parser.add_argument('--model-dir', dest='model-dir',
                        type=str, help='Model dir.')
    parser.add_argument('--bucket', dest='bucket',
                        default="'rare-result-248415-tweet-sentiment-analysis'", type=str, help='bucket name.')
    parser.add_argument('--preprocess-data-dir', dest='preprocess-data-dir', type=str,
                        help='Training data GCS or BQ URI set during Vertex AI training.')
    parser.add_argument('--temp-dir', dest='temp-dir', type=str, help='Temp dir set during Vertex AI training.')
    # Model training args.
    parser.add_argument('--batch-size', dest='batch-size', default=128, type=int,
                        help='Number of examples during each training iteration.')
    parser.add_argument('--load_json', dest='load_json', default='confg.json',
                        help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    hparams = args.__dict__

    trainer.train_evaluate_explain_model(hparams)

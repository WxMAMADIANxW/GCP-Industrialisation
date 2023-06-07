import kfp
from kfp import dsl
import yaml
import json
import kfp.components as comp

with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    PROJECT_ID = cfg['project_id']
    CLIENT_HOST = cfg['client_host']


@dsl.pipeline(
    name='Sentimental analyses Pipeline',
    description='An example pipeline.'
)
def ML_Pipeline():
    """
    Define the Kubeflow pipeline with the Processing step
    """

    preprocess_yaml = 'preprocess.yaml'
    preprocess_config = 'preprocess_config.json'
    train_yaml = 'train.yaml'
    train_config = 'train_config.json'
    test_yaml = 'test.yaml'
    test_config = 'test_config.json'

    _preprocess_op = comp.load_component_from_file(preprocess_yaml)
    with open(preprocess_config, 'r') as f:
        preprocess_params = json.load(f)
        print(preprocess_params)
    preprocess = _preprocess_op(**preprocess_params)

    _train_op = comp.load_component_from_file(train_yaml)
    with open(train_config, 'r') as f:
        train_params = json.load(f)
        print(train_params)
    train = _train_op(preprocess_data_dir=preprocess.output, **train_params)

    _test_op = comp.load_component_from_file(test_yaml)
    with open(test_config, 'r') as f:
        test_params = json.load(f)
        print(test_params)
    test = _test_op(preprocess_data_dir=preprocess.output, model_dir=train.output, **test_params)

    # _preprocess_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # _test_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # _train_op.execution_options.caching_strategy.max_cache_staleness = "P0D"


client = kfp.Client(host=CLIENT_HOST)
client.create_run_from_pipeline_func(ML_Pipeline, arguments={})
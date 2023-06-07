import kfp
from kfp import dsl
import yaml


with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
    PROJECT_ID = cfg['project_id']
    CLIENT_HOST = cfg['client_host']
    PREPROCESS_IMAGE = cfg['preprocess_image']
    TRAIN_IMAGE = cfg['train_image']
    TEST_IMAGE = cfg['test_image']
    INPUT_DATA_URI = cfg['input-data-uri']

client = kfp.Client(host=CLIENT_HOST)

def Preprocess_op():
    """
    Define the preprocessing component
    Returns: to preprocessing component (dsl.ContainerOp)

    """
    return dsl.ContainerOp(
        name='Preprocess Data ',
        image=PREPROCESS_IMAGE,
        arguments=["--input-data-uri", INPUT_DATA_URI],
        file_outputs={'preprocessed-dir': '/Preprocess/preprocess-data-dir.txt'}
    )


def Train_op(preprocess_data_dir : str):
    """
    Define the Training component
    Returns: the Training component (dsl.ContainerOp)

    """
    return dsl.ContainerOp(
        name='Train Model ',
        image=TRAIN_IMAGE,
        arguments=['--preprocess-data-dir', preprocess_data_dir],
        file_outputs={'model-dir': '/trainer/model-dir.txt'}
    )


def Test_op(preprocess_data_dir : str , model_dir):
    """
    Define the Test component
    Returns: the Test component (dsl.ContainerOp)

    """
    return dsl.ContainerOp(
        name='Test Model ',
        image=TEST_IMAGE,
        arguments=[
            '--preprocess-data-dir', preprocess_data_dir,
            '--model-dir', model_dir

        ],
        file_outputs={
            'performance-file': '/test/performance-model.txt'
        }
    )


@dsl.pipeline(
    name='Sentimental analyses Pipeline',
    description='An example pipeline.'
)
def ML_Pipeline():
    """
    Define the Kubeflow pipeline with the Processing step
    """
    _preprocess_op = Preprocess_op()
    _train_op = Train_op(
        _preprocess_op.outputs['preprocessed-dir']
    ).after(_preprocess_op)
    _test_op = Test_op(_preprocess_op.outputs['preprocessed-dir'],_train_op.outputs['model-dir']
    ).after(_train_op)
    _preprocess_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    _test_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    _train_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

client.create_run_from_pipeline_func(ML_Pipeline, arguments={})
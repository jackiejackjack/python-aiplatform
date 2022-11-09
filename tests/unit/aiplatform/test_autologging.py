# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
from importlib import reload
import os
from unittest import mock
from unittest.mock import patch

import pytest
from google.api_core import exceptions
from google.api_core import operation


from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform_v1 import (
    AddContextArtifactsAndExecutionsResponse,
    LineageSubgraph,
    Artifact as GapicArtifact,
    Context as GapicContext,
    Execution as GapicExecution,
    MetadataServiceClient,
    AddExecutionEventsResponse,
    MetadataStore as GapicMetadataStore,
    TensorboardServiceClient,
)
from google.cloud.aiplatform.compat.types import event as gca_event
from google.cloud.aiplatform.compat.types import execution as gca_execution
from google.cloud.aiplatform.compat.types import (
    tensorboard_run as gca_tensorboard_run,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_time_series as gca_tensorboard_time_series,
)
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.metadata import metadata

from google.cloud.aiplatform.compat.services import (
    tensorboard_service_client,
)

from google.cloud.aiplatform.compat.types import (
    encryption_spec as gca_encryption_spec,
    tensorboard as gca_tensorboard,
)

import test_tensorboard
import test_metadata

import numpy as np

_TEST_PROJECT = "test-project"
_TEST_OTHER_PROJECT = "test-project-1"
_TEST_LOCATION = "us-central1"
_TEST_PARENT = (
    f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/metadataStores/default"
)
_TEST_EXPERIMENT = "test-experiment"
_TEST_EXPERIMENT_DESCRIPTION = "test-experiment-description"
_TEST_OTHER_EXPERIMENT_DESCRIPTION = "test-other-experiment-description"
_TEST_PIPELINE = _TEST_EXPERIMENT
_TEST_RUN = "run-1"
_TEST_OTHER_RUN = "run-2"
_TEST_DISPLAY_NAME = "test-display-name"

# resource attributes
_TEST_METADATA = {"test-param1": 1, "test-param2": "test-value", "test-param3": True}

# metadataStore
_TEST_METADATASTORE = (
    f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/metadataStores/default"
)

# context
_TEST_CONTEXT_ID = _TEST_EXPERIMENT
_TEST_CONTEXT_NAME = f"{_TEST_PARENT}/contexts/{_TEST_CONTEXT_ID}"

# execution
_TEST_EXECUTION_ID = f"{_TEST_EXPERIMENT}-{_TEST_RUN}"
_TEST_EXECUTION_NAME = f"{_TEST_PARENT}/executions/{_TEST_EXECUTION_ID}"
_TEST_OTHER_EXECUTION_ID = f"{_TEST_EXPERIMENT}-{_TEST_OTHER_RUN}"
_TEST_OTHER_EXECUTION_NAME = f"{_TEST_PARENT}/executions/{_TEST_OTHER_EXECUTION_ID}"
_TEST_SCHEMA_TITLE = "test.Schema"

_TEST_EXECUTION = GapicExecution(
    name=_TEST_EXECUTION_NAME,
    schema_title=_TEST_SCHEMA_TITLE,
    display_name=_TEST_DISPLAY_NAME,
    metadata=_TEST_METADATA,
    state=GapicExecution.State.RUNNING,
)

# artifact
_TEST_ARTIFACT_ID = f"{_TEST_EXPERIMENT}-{_TEST_RUN}-metrics"
_TEST_ARTIFACT_NAME = f"{_TEST_PARENT}/artifacts/{_TEST_ARTIFACT_ID}"
_TEST_OTHER_ARTIFACT_ID = f"{_TEST_EXPERIMENT}-{_TEST_OTHER_RUN}-metrics"
_TEST_OTHER_ARTIFACT_NAME = f"{_TEST_PARENT}/artifacts/{_TEST_OTHER_ARTIFACT_ID}"

# parameters
_TEST_PARAM_KEY_1 = "learning_rate"
_TEST_PARAM_KEY_2 = "dropout"
_TEST_PARAMS = {_TEST_PARAM_KEY_1: 0.01, _TEST_PARAM_KEY_2: 0.2}
_TEST_OTHER_PARAMS = {_TEST_PARAM_KEY_1: 0.02, _TEST_PARAM_KEY_2: 0.3}

# metrics
_TEST_METRIC_KEY_1 = "rmse"
_TEST_METRIC_KEY_2 = "accuracy"
_TEST_METRICS = {_TEST_METRIC_KEY_1: 222, _TEST_METRIC_KEY_2: 1}
_TEST_OTHER_METRICS = {_TEST_METRIC_KEY_2: 0.9}

# classification_metrics
_TEST_CLASSIFICATION_METRICS = {
    "display_name": "my-classification-metrics",
    "labels": ["cat", "dog"],
    "matrix": [[9, 1], [1, 9]],
    "fpr": [0.1, 0.5, 0.9],
    "tpr": [0.1, 0.7, 0.9],
    "threshold": [0.9, 0.5, 0.1],
}

# schema
_TEST_WRONG_SCHEMA_TITLE = "system.WrongSchema"

# tf model autologging
_TEST_TF_EXPERIMENT_RUN_PARAMS = {
    "batch_size": "None",
    "class_weight": "None",
    "epochs": "5",
    "initial_epoch": "0",
    "max_queue_size": "10",
    "sample_weight": "None",
    "shuffle": "True",
    "steps_per_epoch": "None",
    "use_multiprocessing": "False",
    "validation_batch_size": "None",
    "validation_freq": "1",
    "validation_split": "0.0",
    "validation_steps": "None",
    "workers": "1",
}
_TEST_TF_EXPERIMENT_RUN_METRICS = {
    "accuracy": 0.0,
    "loss": 1.013,
}

# tensorboard
_TEST_TB_ID = "1028944691210842416"
_TEST_TENSORBOARD_NAME = (
    f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/tensorboards/{_TEST_TB_ID}"
)
_TEST_TB_DISPLAY_NAME = "my_tensorboard_1234"
_TEST_ENCRYPTION_KEY_NAME = "key_1234"
_TEST_ENCRYPTION_SPEC = gca_encryption_spec.EncryptionSpec(
    kms_key_name=_TEST_ENCRYPTION_KEY_NAME
)
_TEST_TB_NAME = (
    f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/tensorboards/{_TEST_TB_ID}"
)
_TEST_TENSORBOARD_EXPERIMENT_ID = "test-experiment"
_TEST_TENSORBOARD_EXPERIMENT_NAME = (
    f"{_TEST_TB_NAME}/experiments/{_TEST_TENSORBOARD_EXPERIMENT_ID}"
)

_TEST_TENSORBOARD_RUN_ID = "run-1"
_TEST_TENSORBOARD_RUN_NAME = (
    f"{_TEST_TENSORBOARD_EXPERIMENT_NAME}/runs/{_TEST_TENSORBOARD_RUN_ID}"
)

_TEST_TENSORBOARD_RUN = gca_tensorboard_run.TensorboardRun(
    name=_TEST_TENSORBOARD_RUN_NAME,
    display_name=_TEST_DISPLAY_NAME,
)
_TEST_TIME_SERIES_DISPLAY_NAME = "loss"
_TEST_TIME_SERIES_DISPLAY_NAME_2 = "accuracy"
_TEST_TENSORBOARD_TIME_SERIES_ID = "test-time-series"
_TEST_TENSORBOARD_TIME_SERIES_NAME = (
    f"{_TEST_TENSORBOARD_RUN_NAME}/timeSeries/{_TEST_TENSORBOARD_TIME_SERIES_ID}"
)

_TEST_TENSORBOARD_TIME_SERIES_LIST = [
    gca_tensorboard_time_series.TensorboardTimeSeries(
        name=_TEST_TENSORBOARD_TIME_SERIES_NAME,
        display_name=_TEST_TIME_SERIES_DISPLAY_NAME,
        value_type=gca_tensorboard_time_series.TensorboardTimeSeries.ValueType.SCALAR,
    ),
    gca_tensorboard_time_series.TensorboardTimeSeries(
        name=_TEST_TENSORBOARD_TIME_SERIES_NAME,
        display_name=_TEST_TIME_SERIES_DISPLAY_NAME_2,
        value_type=gca_tensorboard_time_series.TensorboardTimeSeries.ValueType.SCALAR,
    ),
]

# mlflow
_TEST_MLFLOW_TRACKING_URI = "file://my-test-tracking-uri"


@pytest.fixture
def get_tensorboard_mock():
    with patch.object(
        tensorboard_service_client.TensorboardServiceClient, "get_tensorboard"
    ) as get_tensorboard_mock:
        get_tensorboard_mock.return_value = gca_tensorboard.Tensorboard(
            name=_TEST_TENSORBOARD_NAME,
            display_name=_TEST_DISPLAY_NAME,
            encryption_spec=_TEST_ENCRYPTION_SPEC,
        )
        yield get_tensorboard_mock


@pytest.fixture
def get_tensorboard_run_mock():
    with patch.object(
        tensorboard_service_client.TensorboardServiceClient,
        "get_tensorboard_run",
    ) as get_tensorboard_run_mock:
        get_tensorboard_run_mock.return_value = _TEST_TENSORBOARD_RUN
        yield get_tensorboard_run_mock


@pytest.fixture
def create_tensorboard_run_mock():
    with patch.object(
        tensorboard_service_client.TensorboardServiceClient,
        "create_tensorboard_run",
    ) as create_tensorboard_run_mock:
        create_tensorboard_run_mock.return_value = _TEST_TENSORBOARD_RUN
        yield create_tensorboard_run_mock


@pytest.fixture
def list_tensorboard_time_series_mock():
    with patch.object(
        tensorboard_service_client.TensorboardServiceClient,
        "list_tensorboard_time_series",
    ) as list_tensorboard_time_series_mock:
        list_tensorboard_time_series_mock.return_value = (
            _TEST_TENSORBOARD_TIME_SERIES_LIST
        )
        yield list_tensorboard_time_series_mock


create_tensorboard_experiment_mock = test_tensorboard.create_tensorboard_experiment_mock
write_tensorboard_run_data_mock = test_tensorboard.write_tensorboard_run_data_mock
create_tensorboard_time_series_mock = (
    test_tensorboard.create_tensorboard_time_series_mock
)
get_tensorboard_time_series_mock = test_tensorboard.get_tensorboard_time_series_mock
batch_read_tensorboard_time_series_mock = (
    test_tensorboard.batch_read_tensorboard_time_series_mock
)

create_tensorboard_run_artifact_mock = (
    test_metadata.create_tensorboard_run_artifact_mock
)
add_context_artifacts_and_executions_mock = (
    test_metadata.add_context_artifacts_and_executions_mock
)
add_execution_events_mock = test_metadata.add_execution_events_mock


@pytest.fixture
def get_metadata_store_mock():
    with patch.object(
        MetadataServiceClient, "get_metadata_store"
    ) as get_metadata_store_mock:
        get_metadata_store_mock.return_value = GapicMetadataStore(
            name=_TEST_METADATASTORE,
        )
        yield get_metadata_store_mock


@pytest.fixture
def get_metadata_store_mock_raise_not_found_exception():
    with patch.object(
        MetadataServiceClient, "get_metadata_store"
    ) as get_metadata_store_mock:
        get_metadata_store_mock.side_effect = [
            exceptions.NotFound("Test store not found."),
            GapicMetadataStore(
                name=_TEST_METADATASTORE,
            ),
        ]

        yield get_metadata_store_mock


@pytest.fixture
def create_metadata_store_mock():
    with patch.object(
        MetadataServiceClient, "create_metadata_store"
    ) as create_metadata_store_mock:
        create_metadata_store_lro_mock = mock.Mock(operation.Operation)
        create_metadata_store_lro_mock.result.return_value = GapicMetadataStore(
            name=_TEST_METADATASTORE,
        )
        create_metadata_store_mock.return_value = create_metadata_store_lro_mock
        yield create_metadata_store_mock


@pytest.fixture
def get_context_mock():
    with patch.object(MetadataServiceClient, "get_context") as get_context_mock:
        get_context_mock.return_value = GapicContext(
            name=_TEST_CONTEXT_NAME,
            display_name=_TEST_EXPERIMENT,
            description=_TEST_EXPERIMENT_DESCRIPTION,
            schema_title=constants.SYSTEM_EXPERIMENT,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT],
            metadata=constants.EXPERIMENT_METADATA,
        )
        yield get_context_mock


@pytest.fixture
def get_context_wrong_schema_mock():
    with patch.object(
        MetadataServiceClient, "get_context"
    ) as get_context_wrong_schema_mock:
        get_context_wrong_schema_mock.return_value = GapicContext(
            name=_TEST_CONTEXT_NAME,
            display_name=_TEST_EXPERIMENT,
            schema_title=_TEST_WRONG_SCHEMA_TITLE,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT],
            metadata=constants.EXPERIMENT_METADATA,
        )
        yield get_context_wrong_schema_mock


@pytest.fixture
def get_pipeline_context_mock():
    with patch.object(
        MetadataServiceClient, "get_context"
    ) as get_pipeline_context_mock:
        get_pipeline_context_mock.return_value = GapicContext(
            name=_TEST_CONTEXT_NAME,
            display_name=_TEST_EXPERIMENT,
            schema_title=constants.SYSTEM_PIPELINE,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_PIPELINE],
            metadata=constants.EXPERIMENT_METADATA,
        )
        yield get_pipeline_context_mock


@pytest.fixture
def get_context_not_found_mock():
    with patch.object(
        MetadataServiceClient, "get_context"
    ) as get_context_not_found_mock:
        get_context_not_found_mock.side_effect = exceptions.NotFound("test: not found")
        yield get_context_not_found_mock


_TEST_EXPERIMENT_CONTEXT = GapicContext(
    name=_TEST_CONTEXT_NAME,
    display_name=_TEST_EXPERIMENT,
    description=_TEST_EXPERIMENT_DESCRIPTION,
    schema_title=constants.SYSTEM_EXPERIMENT,
    schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT],
    metadata={
        **constants.EXPERIMENT_METADATA,
        constants._BACKING_TENSORBOARD_RESOURCE_KEY: test_tensorboard._TEST_NAME,
    },
)


@pytest.fixture
def add_context_children_mock():
    with patch.object(
        MetadataServiceClient, "add_context_children"
    ) as add_context_children_mock:
        yield add_context_children_mock


@pytest.fixture
def add_context_artifacts_and_executions_mock():
    with patch.object(
        MetadataServiceClient, "add_context_artifacts_and_executions"
    ) as add_context_artifacts_and_executions_mock:
        add_context_artifacts_and_executions_mock.return_value = (
            AddContextArtifactsAndExecutionsResponse()
        )
        yield add_context_artifacts_and_executions_mock


@pytest.fixture
def get_execution_mock():
    with patch.object(MetadataServiceClient, "get_execution") as get_execution_mock:
        get_execution_mock.return_value = GapicExecution(
            name=_TEST_EXECUTION_NAME,
            display_name=_TEST_RUN,
            schema_title=constants.SYSTEM_RUN,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
        )
        yield get_execution_mock


@pytest.fixture
def get_execution_wrong_schema_mock():
    with patch.object(
        MetadataServiceClient, "get_execution"
    ) as get_execution_wrong_schema_mock:
        get_execution_wrong_schema_mock.return_value = GapicExecution(
            name=_TEST_EXECUTION_NAME,
            display_name=_TEST_RUN,
            schema_title=_TEST_WRONG_SCHEMA_TITLE,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
        )
        yield get_execution_wrong_schema_mock


@pytest.fixture
def update_execution_mock():
    with patch.object(
        MetadataServiceClient, "update_execution"
    ) as update_execution_mock:
        update_execution_mock.return_value = GapicExecution(
            name=_TEST_EXECUTION_NAME,
            display_name=_TEST_RUN,
            schema_title=constants.SYSTEM_RUN,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
            metadata=_TEST_PARAMS,
        )
        yield update_execution_mock


@pytest.fixture
def add_execution_events_mock():
    with patch.object(
        MetadataServiceClient, "add_execution_events"
    ) as add_execution_events_mock:
        add_execution_events_mock.return_value = AddExecutionEventsResponse()
        yield add_execution_events_mock


@pytest.fixture
def list_executions_mock():
    with patch.object(MetadataServiceClient, "list_executions") as list_executions_mock:
        list_executions_mock.return_value = [
            GapicExecution(
                name=_TEST_EXECUTION_NAME,
                display_name=_TEST_RUN,
                schema_title=constants.SYSTEM_RUN,
                schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
                metadata=_TEST_PARAMS,
            ),
            GapicExecution(
                name=_TEST_OTHER_EXECUTION_NAME,
                display_name=_TEST_OTHER_RUN,
                schema_title=constants.SYSTEM_RUN,
                schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_RUN],
                metadata=_TEST_OTHER_PARAMS,
            ),
        ]
        yield list_executions_mock


@pytest.fixture
def get_tensorboard_run_not_found_mock():
    with patch.object(
        TensorboardServiceClient, "get_tensorboard_run"
    ) as get_tensorboard_run_mock:
        get_tensorboard_run_mock.side_effect = [
            exceptions.NotFound(""),
            test_tensorboard._TEST_TENSORBOARD_RUN,
        ]
        yield get_tensorboard_run_mock


@pytest.fixture
def get_tensorboard_experiment_not_found_mock():
    with patch.object(
        TensorboardServiceClient, "get_tensorboard_experiment"
    ) as get_tensorboard_experiment_mock:
        get_tensorboard_experiment_mock.side_effect = [
            exceptions.NotFound(""),
            test_tensorboard._TEST_TENSORBOARD_EXPERIMENT,
        ]
        yield get_tensorboard_experiment_mock


@pytest.fixture
def get_tensorboard_time_series_not_found_mock():
    with patch.object(
        TensorboardServiceClient, "get_tensorboard_time_series"
    ) as get_tensorboard_time_series_mock:
        get_tensorboard_time_series_mock.side_effect = [
            exceptions.NotFound(""),
            # test_tensorboard._TEST_TENSORBOARD_TIME_SERIES # change to time series
        ]
        yield get_tensorboard_time_series_mock


@pytest.fixture
def query_execution_inputs_and_outputs_mock():
    with patch.object(
        MetadataServiceClient, "query_execution_inputs_and_outputs"
    ) as query_execution_inputs_and_outputs_mock:
        query_execution_inputs_and_outputs_mock.side_effect = [
            LineageSubgraph(
                artifacts=[
                    GapicArtifact(
                        name=_TEST_ARTIFACT_NAME,
                        display_name=_TEST_ARTIFACT_ID,
                        schema_title=constants.SYSTEM_METRICS,
                        schema_version=constants.SCHEMA_VERSIONS[
                            constants.SYSTEM_METRICS
                        ],
                        metadata=_TEST_METRICS,
                    )
                ],
                events=[
                    gca_event.Event(
                        artifact=_TEST_ARTIFACT_NAME,
                        execution=_TEST_EXECUTION_NAME,
                        type_=gca_event.Event.Type.OUTPUT,
                    )
                ],
            ),
            LineageSubgraph(
                artifacts=[
                    GapicArtifact(
                        name=_TEST_OTHER_ARTIFACT_NAME,
                        display_name=_TEST_OTHER_ARTIFACT_ID,
                        schema_title=constants.SYSTEM_METRICS,
                        schema_version=constants.SCHEMA_VERSIONS[
                            constants.SYSTEM_METRICS
                        ],
                        metadata=_TEST_OTHER_METRICS,
                    ),
                ],
                events=[
                    gca_event.Event(
                        artifact=_TEST_OTHER_ARTIFACT_NAME,
                        execution=_TEST_OTHER_EXECUTION_NAME,
                        type_=gca_event.Event.Type.OUTPUT,
                    )
                ],
            ),
        ]
        yield query_execution_inputs_and_outputs_mock


_TEST_CLASSIFICATION_METRICS_METADATA = {
    "confusionMatrix": {
        "annotationSpecs": [{"displayName": "cat"}, {"displayName": "dog"}],
        "rows": [[9, 1], [1, 9]],
    },
    "confidenceMetrics": [
        {"confidenceThreshold": 0.9, "recall": 0.1, "falsePositiveRate": 0.1},
        {"confidenceThreshold": 0.5, "recall": 0.7, "falsePositiveRate": 0.5},
        {"confidenceThreshold": 0.1, "recall": 0.9, "falsePositiveRate": 0.9},
    ],
}

_TEST_CLASSIFICATION_METRICS_ARTIFACT = GapicArtifact(
    name=_TEST_ARTIFACT_NAME,
    display_name=_TEST_CLASSIFICATION_METRICS["display_name"],
    schema_title=constants.GOOGLE_CLASSIFICATION_METRICS,
    schema_version=constants._DEFAULT_SCHEMA_VERSION,
    metadata=_TEST_CLASSIFICATION_METRICS_METADATA,
    state=GapicArtifact.State.LIVE,
)


@pytest.fixture
def create_classification_metrics_artifact_mock():
    with patch.object(
        MetadataServiceClient, "create_artifact"
    ) as create_classification_metrics_artifact_mock:
        create_classification_metrics_artifact_mock.return_value = (
            _TEST_CLASSIFICATION_METRICS_ARTIFACT
        )
        yield create_classification_metrics_artifact_mock


@pytest.fixture
def get_classification_metrics_artifact_mock():
    with patch.object(
        MetadataServiceClient, "get_artifact"
    ) as get_classification_metrics_artifact_mock:
        get_classification_metrics_artifact_mock.return_value = (
            _TEST_CLASSIFICATION_METRICS_ARTIFACT
        )
        yield get_classification_metrics_artifact_mock


@pytest.fixture
def get_artifact_mock():
    with patch.object(MetadataServiceClient, "get_artifact") as get_artifact_mock:
        get_artifact_mock.return_value = GapicArtifact(
            name=_TEST_ARTIFACT_NAME,
            display_name=_TEST_ARTIFACT_ID,
            schema_title=constants.SYSTEM_METRICS,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_METRICS],
        )
        yield get_artifact_mock


@pytest.fixture
def get_artifact_not_found_mock():
    with patch.object(MetadataServiceClient, "get_artifact") as get_artifact_mock:
        get_artifact_mock.side_effect = exceptions.NotFound("")
        yield get_artifact_mock


@pytest.fixture
def get_artifact_wrong_schema_mock():
    with patch.object(
        MetadataServiceClient, "get_artifact"
    ) as get_artifact_wrong_schema_mock:
        get_artifact_wrong_schema_mock.return_value = GapicArtifact(
            name=_TEST_ARTIFACT_NAME,
            display_name=_TEST_ARTIFACT_ID,
            schema_title=_TEST_WRONG_SCHEMA_TITLE,
            schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_METRICS],
        )
        yield get_artifact_wrong_schema_mock


@pytest.fixture
def update_context_mock():
    with patch.object(MetadataServiceClient, "update_context") as update_context_mock:
        update_context_mock.return_value = _EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT
        yield update_context_mock


_TEST_EXPERIMENT_RUN_CONTEXT_NAME = f"{_TEST_PARENT}/contexts/{_TEST_EXECUTION_ID}"
_TEST_OTHER_EXPERIMENT_RUN_CONTEXT_NAME = (
    f"{_TEST_PARENT}/contexts/{_TEST_OTHER_EXECUTION_ID}"
)

_EXPERIMENT_MOCK = GapicContext(
    name=_TEST_CONTEXT_NAME,
    display_name=_TEST_EXPERIMENT,
    description=_TEST_EXPERIMENT_DESCRIPTION,
    schema_title=constants.SYSTEM_EXPERIMENT,
    schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT],
    metadata={
        **constants.EXPERIMENT_METADATA,
        constants._BACKING_TENSORBOARD_RESOURCE_KEY: test_tensorboard._TEST_NAME,
    },
)

_EXPERIMENT_RUN_MOCK = GapicContext(
    name=_TEST_EXPERIMENT_RUN_CONTEXT_NAME,
    display_name=_TEST_RUN,
    schema_title=constants.SYSTEM_EXPERIMENT_RUN,
    schema_version=constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT_RUN],
    metadata={
        constants._PARAM_KEY: _TEST_TF_EXPERIMENT_RUN_PARAMS,
        constants._METRIC_KEY: _TEST_TF_EXPERIMENT_RUN_METRICS,
        constants._STATE_KEY: gca_execution.Execution.State.RUNNING.name,
    },
)

_EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT = copy.deepcopy(_EXPERIMENT_RUN_MOCK)
_EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT.parent_contexts = [_TEST_CONTEXT_NAME]


@pytest.fixture
def get_experiment_mock():
    with patch.object(MetadataServiceClient, "get_context") as get_context_mock:
        get_context_mock.return_value = _EXPERIMENT_MOCK
        yield get_context_mock


@pytest.fixture
def get_experiment_run_run_mock():
    with patch.object(MetadataServiceClient, "get_context") as get_context_mock:
        get_context_mock.side_effect = [
            _EXPERIMENT_MOCK,
            _EXPERIMENT_RUN_MOCK,
            _EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT,
        ]

        yield get_context_mock


@pytest.fixture
def get_experiment_run_mock():
    with patch.object(MetadataServiceClient, "get_context") as get_context_mock:
        get_context_mock.side_effect = [
            _EXPERIMENT_MOCK,
            _EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT,
        ]

        yield get_context_mock


@pytest.fixture
def create_experiment_context_mock():
    with patch.object(MetadataServiceClient, "create_context") as create_context_mock:
        create_context_mock.side_effect = [_TEST_EXPERIMENT_CONTEXT]
        yield create_context_mock


@pytest.fixture
def create_experiment_run_context_mock():
    with patch.object(MetadataServiceClient, "create_context") as create_context_mock:
        create_context_mock.side_effect = [_EXPERIMENT_RUN_MOCK]
        yield create_context_mock


def build_and_train_test_tf_model():
    import tensorflow as tf

    X = np.array(
        [
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 3],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 3],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 3],
        ]
    )
    y = np.dot(X, np.array([1, 2])) + 3

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(2,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(X, y, epochs=5)


@pytest.mark.usefixtures("google_auth_mock")
class TestAutologging:
    def setup_method(self):
        reload(initializer)
        reload(metadata)
        reload(aiplatform)

    def teardown_method(self):
        initializer.global_pool.shutdown(wait=True)

    @pytest.mark.usefixtures(
        "get_experiment_mock",
        "get_metadata_store_mock",
        "create_experiment_run_context_mock",
    )
    def test_autologging_init(
        self,
        create_experiment_run_context_mock,
    ):

        try:
            import packaging
            import mlflow  # noqa: F401
        except ImportError:
            raise ImportError(
                "MLFlow is not installed and is required to test autologging. "
                'Please install the SDK using "pip install google-cloud-aiplatform[autologging]"'
            )
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise ImportError(
                "TensorFlow is not installed and is required to test autologging."
                'Please install it before running autologging tests."'
            )
        aiplatform.init(
            project=_TEST_PROJECT, location=_TEST_LOCATION, experiment=_TEST_EXPERIMENT
        )

        aiplatform.autolog()

    def test_autologging_raises_if_experiment_not_set(
        self,
    ):
        aiplatform.init(project=_TEST_PROJECT, location=_TEST_LOCATION)

        with pytest.raises(ValueError):
            aiplatform.autolog()

    @pytest.mark.usefixtures(
        "get_metadata_store_mock",
        "get_experiment_run_run_mock",
    )
    def test_autologging_sets_and_resets_mlflow_tracking_uri(
        self,
    ):
        import mlflow

        aiplatform.init(
            project=_TEST_PROJECT, location=_TEST_LOCATION, experiment=_TEST_EXPERIMENT
        )
        mlflow.set_tracking_uri(_TEST_MLFLOW_TRACKING_URI)

        aiplatform.autolog()

        assert mlflow.get_tracking_uri() == "vertex-mlflow-plugin://"

        aiplatform.autolog(disable=True)

        assert mlflow.get_tracking_uri() == _TEST_MLFLOW_TRACKING_URI

    @pytest.mark.usefixtures(
        "get_metadata_store_mock",
        "create_experiment_run_context_mock",
        "add_context_children_mock",
        "get_experiment_mock",
        "get_experiment_run_run_mock",
        "get_tensorboard_mock",
        "create_tensorboard_experiment_mock",
        "write_tensorboard_run_data_mock",
        "get_tensorboard_experiment_not_found_mock",
        "get_artifact_not_found_mock",
        "get_tensorboard_time_series_not_found_mock",
        "list_tensorboard_time_series_mock",
        "create_tensorboard_run_artifact_mock",
        "get_tensorboard_time_series_mock",
        "get_tensorboard_run_mock",
        "update_context_mock",
    )
    def test_autologging_with_auto_run_creation(
        self,
        get_experiment_mock,
        get_metadata_store_mock,
        get_experiment_run_mock,
        get_tensorboard_mock,
        create_experiment_run_context_mock,
        add_context_artifacts_and_executions_mock,
        write_tensorboard_run_data_mock,
        update_context_mock,
    ):
        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            experiment=_TEST_EXPERIMENT,
        )

        aiplatform.autolog()

        build_and_train_test_tf_model()

        assert create_experiment_run_context_mock.call_count == 1
        assert get_metadata_store_mock.call_count == 1
        assert (
            write_tensorboard_run_data_mock.call_count == 4
        )  # TODO: document why this is called 4 times
        assert get_tensorboard_mock.call_count == 1

        _TRUE_CONTEXT = copy.deepcopy(_EXPERIMENT_RUN_MOCK_WITH_PARENT_EXPERIMENT)
        _TRUE_CONTEXT.metadata[
            constants._STATE_KEY
        ] = gca_execution.Execution.State.COMPLETE.name

        update_context_mock.assert_called_with(context=_TRUE_CONTEXT)

        for args, kwargs in create_experiment_run_context_mock.call_args_list:
            assert kwargs["context"].display_name.startswith("tensorflow")
            assert kwargs["context_id"].startswith(f"{_TEST_EXPERIMENT}-tensorflow")

        build_and_train_test_tf_model()

        # a subsequent model.fit() call should create another ExperimentRun
        assert create_experiment_run_context_mock.call_count == 2

        # the above model.fit() calls should not result in any data being written locally
        assert not os.path.isdir("mlruns")

        # training a model after disabling autologging should not create additional ExperimentRuns
        aiplatform.autolog(disable=True)
        build_and_train_test_tf_model()
        assert create_experiment_run_context_mock.call_count == 2

        if os.path.exists(os.path.join(os.getcwd(), "mlruns")) and os.access(
            os.path.join(os.getcwd(), "mlruns"), os.W_OK
        ):
            assert os.path.isdir("mlruns")

    @pytest.mark.usefixtures(
        "get_metadata_store_mock",
        "create_experiment_run_context_mock",
        "add_context_children_mock",
        "get_experiment_mock",
        "get_experiment_run_run_mock",
        "get_tensorboard_mock",
        "create_tensorboard_experiment_mock",
        "write_tensorboard_run_data_mock",
        "get_tensorboard_experiment_not_found_mock",
        "get_artifact_not_found_mock",
        "get_tensorboard_time_series_not_found_mock",
        "list_tensorboard_time_series_mock",
        "create_tensorboard_run_artifact_mock",
        "get_tensorboard_time_series_mock",
        "get_tensorboard_run_mock",
        "update_context_mock",
    )
    def test_autologging_with_manual_run_creation(
        self,
        get_experiment_mock,
        get_metadata_store_mock,
        get_experiment_run_mock,
        get_tensorboard_mock,
        create_experiment_run_context_mock,
        add_context_artifacts_and_executions_mock,
        write_tensorboard_run_data_mock,
        update_context_mock,
    ):
        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            experiment=_TEST_EXPERIMENT,
        )

        aiplatform.autolog()

        aiplatform.start_run(_TEST_RUN)
        build_and_train_test_tf_model()
        assert create_experiment_run_context_mock.call_count == 1

        # training an additional model after calling start_run() should not result in a new ExperimentRun
        build_and_train_test_tf_model()
        assert create_experiment_run_context_mock.call_count == 1

        # ending the run and training a new model should result in an auto-created run
        aiplatform.end_run()
        build_and_train_test_tf_model()
        assert create_experiment_run_context_mock.call_count == 2

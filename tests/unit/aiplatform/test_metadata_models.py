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

import datetime
from importlib import reload
from unittest import mock
from unittest.mock import patch

from google.auth import credentials as auth_credentials
from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.metadata import metadata
from google.cloud.aiplatform_v1 import Artifact as GapicArtifact
from google.cloud.aiplatform_v1 import MetadataServiceClient
import numpy as np
import pytest
import sklearn
from sklearn.linear_model import LinearRegression


# project
_TEST_PROJECT = "test-project"
_TEST_LOCATION = "us-central1"
_TEST_BUCKET_NAME = "gs://test-bucket"
_TEST_PARENT = (
    f"projects/{_TEST_PROJECT}/locations/{_TEST_LOCATION}/metadataStores/default"
)
_TEST_CREDENTIALS = mock.Mock(spec=auth_credentials.AnonymousCredentials())


# artifact
_TEST_ARTIFACT_ID = "test-model-id"
_TEST_URI = "gs://test-uri"
_TEST_DISPLAY_NAME = "test-model-display-name"

_TEST_ARTIFACT_ID = "test-model-id"
_TEST_ARTIFACT_NAME = f"{_TEST_PARENT}/artifacts/{_TEST_ARTIFACT_ID}"

_TEST_TIMESTAMP = "20221130000000"
_TEST_DATETIME = datetime.datetime.strptime(_TEST_TIMESTAMP, "%Y%m%d%H%M%S")

_TEST_INPUT_EXAMPLE = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])


@pytest.fixture
def mock_datetime_now(monkeypatch):
    class DateTime:
        @classmethod
        def now(cls):
            return _TEST_DATETIME

    monkeypatch.setattr(datetime, "datetime", DateTime)


@pytest.fixture
def mock_storage_blob_upload_from_filename():
    with patch(
        "google.cloud.storage.Blob.upload_from_filename"
    ) as mock_blob_upload_from_filename, patch(
        "google.cloud.storage.Bucket.exists", return_value=True
    ):
        yield mock_blob_upload_from_filename


_TEST_EXPERIMENT_MODEL_ARTIFACT = GapicArtifact(
    name=_TEST_ARTIFACT_NAME,
    display_name=_TEST_DISPLAY_NAME,
    schema_title=constants.GOOGLE_EXPERIMENT_MODEL,
    schema_version=constants.DEFAULT_SCHEMA_VERSION,
    state=GapicArtifact.State.LIVE,
)


@pytest.fixture
def create_experiment_model_artifact_mock():
    with patch.object(
        MetadataServiceClient, "create_artifact"
    ) as create_experiment_model_artifact_mock:
        create_experiment_model_artifact_mock.return_value = (
            _TEST_EXPERIMENT_MODEL_ARTIFACT
        )
        yield create_experiment_model_artifact_mock


@pytest.fixture
def get_experiment_model_artifact_mock():
    with patch.object(
        MetadataServiceClient, "get_artifact"
    ) as get_experiment_model_artifact_mock:
        get_experiment_model_artifact_mock.return_value = (
            _TEST_EXPERIMENT_MODEL_ARTIFACT
        )
        yield get_experiment_model_artifact_mock


class TestModels:
    def setup_method(self):
        reload(initializer)
        reload(metadata)
        reload(aiplatform)

    def teardown_method(self):
        initializer.global_pool.shutdown(wait=True)

    @pytest.mark.usefixtures(
        "mock_datetime_now",
    )
    def test_save_model_sklearn(
        self,
        mock_storage_blob_upload_from_filename,
        create_experiment_model_artifact_mock,
        get_experiment_model_artifact_mock,
    ):
        train_x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        train_y = np.dot(train_x, np.array([1, 2])) + 3
        model = LinearRegression()
        model.fit(train_x, train_y)

        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            staging_bucket=_TEST_BUCKET_NAME,
            credentials=_TEST_CREDENTIALS,
        )

        aiplatform.save_model(model, _TEST_ARTIFACT_ID)

        # Verify that the model file is correctly uploaded to gcs
        upload_file_path = mock_storage_blob_upload_from_filename.call_args[1][
            "filename"
        ]
        assert upload_file_path.endswith("model.pkl")

        # Verify the model artifact is created correctly
        expected_artifact = GapicArtifact(
            uri=f"{_TEST_BUCKET_NAME}/sklearn-model-{_TEST_TIMESTAMP}",
            schema_title=constants.GOOGLE_EXPERIMENT_MODEL,
            schema_version=constants.DEFAULT_SCHEMA_VERSION,
            metadata={
                "frameworkName": "sklearn",
                "frameworkVersion": sklearn.__version__,
                "modelFile": "model.pkl",
                "modelClass": "LinearRegression",
            },
            state=GapicArtifact.State.LIVE,
        )
        create_experiment_model_artifact_mock.assert_called_once_with(
            parent=_TEST_PARENT,
            artifact=expected_artifact,
            artifact_id=_TEST_ARTIFACT_ID,
        )

        get_experiment_model_artifact_mock.assert_called_once_with(
            name=_TEST_ARTIFACT_NAME, retry=base._DEFAULT_RETRY
        )

    @pytest.mark.usefixtures(
        "mock_storage_blob_upload_from_filename",
        "get_experiment_model_artifact_mock",
    )
    def test_save_model_with_all_args(
        self,
        create_experiment_model_artifact_mock,
    ):
        train_x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        train_y = np.dot(train_x, np.array([1, 2])) + 3
        model = LinearRegression()
        model.fit(train_x, train_y)

        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            staging_bucket=_TEST_BUCKET_NAME,
            credentials=_TEST_CREDENTIALS,
        )

        aiplatform.save_model(
            model=model,
            artifact_id=_TEST_ARTIFACT_ID,
            uri=_TEST_URI,
            display_name=_TEST_DISPLAY_NAME,
            input_example=_TEST_INPUT_EXAMPLE,
        )

        # Verify the model artifact is created correctly
        expected_artifact = GapicArtifact(
            display_name=_TEST_DISPLAY_NAME,
            uri=_TEST_URI,
            schema_title=constants.GOOGLE_EXPERIMENT_MODEL,
            schema_version=constants.DEFAULT_SCHEMA_VERSION,
            metadata={
                "frameworkName": "sklearn",
                "frameworkVersion": sklearn.__version__,
                "modelFile": "model.pkl",
                "modelClass": "LinearRegression",
                "predictSchemata": {"instanceSchemaUri": f"{_TEST_URI}/instance.yaml"},
            },
            state=GapicArtifact.State.LIVE,
        )
        create_experiment_model_artifact_mock.assert_called_once_with(
            parent=_TEST_PARENT,
            artifact=expected_artifact,
            artifact_id=_TEST_ARTIFACT_ID,
        )

    @pytest.mark.usefixtures(
        "mock_storage_blob_upload_from_filename",
        "create_experiment_model_artifact_mock",
        "get_experiment_model_artifact_mock",
    )
    def test_save_model_without_staging_bucket_error(self):
        train_x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        train_y = np.dot(train_x, np.array([1, 2])) + 3
        model = LinearRegression()
        model.fit(train_x, train_y)

        aiplatform.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
            credentials=_TEST_CREDENTIALS,
        )

        with pytest.raises(ValueError) as exception:
            aiplatform.save_model(model=model, artifact_id=_TEST_ARTIFACT_ID)
        assert str(exception.value) == (
            "No staging bucket set. Make sure to call "
            + "aiplatform.init(staging_bucket='gs://my-bucket') "
            + "or specify the 'uri' when saving the model. "
        )

    def test_load_model_sklearn(self):
        pass

    def test_register_model_sklearn(self):
        pass

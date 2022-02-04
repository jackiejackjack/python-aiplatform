# -*- coding: utf-8 -*-
# Copyright 2020 Google LLC
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
# Generated code. DO NOT EDIT!
#
# Snippet for ImportFeatureValues
# NOTE: This snippet has been automatically generated for illustrative purposes only.
# It may require modifications to work in your environment.

# To install the latest published package dependency, execute the following:
#   python3 -m pip install google-cloud-aiplatform


# [START aiplatform_generated_aiplatform_v1_FeaturestoreService_ImportFeatureValues_async]
from google.cloud import aiplatform_v1


async def sample_import_feature_values():
    # Create a client
    client = aiplatform_v1.FeaturestoreServiceAsyncClient()

    # Initialize request argument(s)
    avro_source = aiplatform_v1.AvroSource()
    avro_source.gcs_source.uris = ['uris_value_1', 'uris_value_2']

    feature_specs = aiplatform_v1.FeatureSpec()
    feature_specs.id = "id_value"

    request = aiplatform_v1.ImportFeatureValuesRequest(
        avro_source=avro_source,
        feature_time_field="feature_time_field_value",
        entity_type="entity_type_value",
        feature_specs=feature_specs,
    )

    # Make the request
    operation = client.import_feature_values(request=request)

    print("Waiting for operation to complete...")

    response = await operation.result()
    print(response)

# [END aiplatform_generated_aiplatform_v1_FeaturestoreService_ImportFeatureValues_async]

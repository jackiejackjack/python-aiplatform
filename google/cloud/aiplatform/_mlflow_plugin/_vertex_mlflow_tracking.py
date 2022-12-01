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

import packaging
from mlflow.store.tracking.abstract_store import AbstractStore

from mlflow.entities import (
    Experiment,
    RunTag,
    Metric,
    Param,
    Run,
    RunInfo,
    RunData,
    LifecycleStage,
    ViewType,
)

import uuid
from google.cloud import aiplatform

from typing import List, Optional, Tuple


class _VertexMlflowTracking(AbstractStore):
    """Vertex plugin implementation of MLFlow's AbstractStore class."""

    def to_mlflow_metric(self, vertex_metrics) -> Optional[Metric]:

        mlflow_metrics = []

        if vertex_metrics == {}:
            return None
        else:
            for metric_key in vertex_metrics:
                mlflow_metric = Metric(
                    key=metric_key,
                    value=vertex_metrics[metric_key],
                    step=0,
                    timestamp=0,
                )
                mlflow_metrics.append(mlflow_metric)

        return mlflow_metrics

    def to_mlflow_params(self, vertex_params) -> Optional[Param]:
        mlflow_params = []

        if vertex_params == {}:
            return None
        else:
            for param_key in vertex_params:
                mlflow_param = Param(key=param_key, value=vertex_params[param_key])
                mlflow_params.append(mlflow_param)

        return mlflow_params

    def to_mlflow_entity(
        self,
        vertex_exp: "aiplatform.Experiment",
        vertex_run: "aiplatform.ExperimentRun",
    ) -> Run:

        run_info = RunInfo(
            run_uuid=vertex_run.name,
            run_id=vertex_run.name,
            experiment_id=vertex_exp.name,
            user_id="",
            status=vertex_run.state,
            start_time=1,
            end_time=2,
            lifecycle_stage=LifecycleStage.ACTIVE,
            artifact_uri="file:///tmp/my_tracking",
        )

        run_data = RunData(
            metrics=self.to_mlflow_metric(vertex_run.get_metrics()),
            params=self.to_mlflow_params(vertex_run.get_params()),
            tags={},
        )

        return Run(run_info=run_info, run_data=run_data)

    def __init__(self, store_uri: str = None, artifact_uri: str = None) -> None:
        super(_VertexMlflowTracking, self).__init__()

    def create_run(
        self, experiment_id, user_id, start_time, tags: List[RunTag], run_name
    ) -> Run:
        run_uuid = f"{uuid.uuid4()}"
        self.run_id = run_uuid

        framework = ""

        for tag in tags:
            if tag.key == "mlflow.autologging":
                framework = tag.value

        vertex_run_id = f"{framework}-{run_uuid}"

        self.vertex_experiment = (
            aiplatform.metadata.metadata._experiment_tracker._experiment
        )
        self.vertex_experiment_run = (
            aiplatform.metadata.metadata._experiment_tracker._experiment_run
        )

        if self.vertex_experiment_run:
            self.autocreate = False

        # Create a new run for the user if they haven't called aiplatform.start_run()
        if self.vertex_experiment_run is None:
            print("autocreating a run")
            self.autocreate = True
            new_vertex_run = vertex_run_id
            self.vertex_experiment_run = aiplatform.start_run(run=new_vertex_run)

        return self.to_mlflow_entity(
            vertex_exp=self.vertex_experiment, vertex_run=self.vertex_experiment_run
        )

    def update_run_info(self, run_id, run_status, end_time, run_name) -> RunInfo:
        print("in update run info")
        print(run_status, "run_status", self.autocreate)

        # a run_status of 3 means the run has finished
        # see here: https://www.mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunStatus
        if run_status == 3 and self.autocreate:
            print("ending run")
            aiplatform.end_run()
            self.vertex_experiment_run = None

        return RunInfo(
            run_uuid=run_id,
            status=run_status,
            end_time=end_time,
            experiment_id=self.vertex_experiment,
            user_id="",
            start_time=1,
            lifecycle_stage=LifecycleStage.ACTIVE,
        )

    def log_batch(
        self,
        run_id: str,
        metrics: List[Metric],
        params: List[Param],
        tags: List[RunTag],
    ) -> None:

        import datetime

        print("in log batch", str(datetime.datetime.now()))

        summary_metrics = {}
        summary_params = {}
        time_series_metrics = {}

        for metric in metrics:
            if metric.step:
                if metric.step not in time_series_metrics:
                    time_series_metrics[metric.step] = {metric.key: metric.value}
                else:
                    time_series_metrics[metric.step][metric.key] = metric.value
            else:
                summary_metrics[metric.key] = metric.value

        for param in params:
            summary_params[param.key] = param.value

        if summary_metrics:
            aiplatform.log_metrics(metrics=summary_metrics)

        if summary_params:
            aiplatform.log_params(params=summary_params)

        if time_series_metrics:
            for step in time_series_metrics:
                aiplatform.log_time_series_metrics(time_series_metrics[step], step)

    def log_metric(self, run_id: str, metric: Metric) -> None:
        print("in log metric", metric)
        self.log_batch(run_id, metric, params={})

    def list_experiments(
        self, view_type: str = ViewType.ACTIVE_ONLY
    ) -> List[Experiment]:
        pass

    def create_experiment(
        self, name: str, artifact_location: str, tags: List[RunTag]
    ) -> str:
        pass

    def get_experiment(self, experiment_id: str) -> Experiment:
        pass

    def delete_experiment(self, experiment_id: str) -> None:
        pass

    def restore_experiment(self, experiment_id: str) -> None:
        pass

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        pass

    def get_run(self, run_id: str) -> Run:
        print("in get run")
        return self.to_mlflow_entity(
            vertex_exp=self.vertex_experiment, vertex_run=self.vertex_experiment_run
        )

    def restore_run(self, run_id: str) -> None:
        pass

    def get_metric_history(self, run_id: str, metric_key: str) -> List[Metric]:
        pass

    def _search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str,
        run_view_type: str,
        max_results: int,
        order_by: List[str],
        page_token: str = None,
    ) -> Tuple[List[Run], str]:
        pass

    def delete_run(self, run_id: str) -> None:
        pass

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
from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package='google.cloud.aiplatform.v1',
    manifest={
        'Study',
        'Trial',
        'StudySpec',
        'Measurement',
    },
)


class Study(proto.Message):
    r"""A message representing a Study.

    Attributes:
        name (str):
            Output only. The name of a study. The study's globally
            unique identifier. Format:
            ``projects/{project}/locations/{location}/studies/{study}``
        display_name (str):
            Required. Describes the Study, default value
            is empty string.
        study_spec (google.cloud.aiplatform_v1.types.StudySpec):
            Required. Configuration of the Study.
        state (google.cloud.aiplatform_v1.types.Study.State):
            Output only. The detailed state of a Study.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time at which the study was
            created.
        inactive_reason (str):
            Output only. A human readable reason why the
            Study is inactive. This should be empty if a
            study is ACTIVE or COMPLETED.
    """
    class State(proto.Enum):
        r"""Describes the Study state."""
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2
        COMPLETED = 3

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    study_spec: 'StudySpec' = proto.Field(
        proto.MESSAGE,
        number=3,
        message='StudySpec',
    )
    state: State = proto.Field(
        proto.ENUM,
        number=4,
        enum=State,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    inactive_reason: str = proto.Field(
        proto.STRING,
        number=6,
    )


class Trial(proto.Message):
    r"""A message representing a Trial. A Trial contains a unique set
    of Parameters that has been or will be evaluated, along with the
    objective metrics got by running the Trial.

    Attributes:
        name (str):
            Output only. Resource name of the Trial
            assigned by the service.
        id (str):
            Output only. The identifier of the Trial
            assigned by the service.
        state (google.cloud.aiplatform_v1.types.Trial.State):
            Output only. The detailed state of the Trial.
        parameters (MutableSequence[google.cloud.aiplatform_v1.types.Trial.Parameter]):
            Output only. The parameters of the Trial.
        final_measurement (google.cloud.aiplatform_v1.types.Measurement):
            Output only. The final measurement containing
            the objective value.
        measurements (MutableSequence[google.cloud.aiplatform_v1.types.Measurement]):
            Output only. A list of measurements that are strictly
            lexicographically ordered by their induced tuples (steps,
            elapsed_duration). These are used for early stopping
            computations.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the Trial was started.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the Trial's status changed to
            ``SUCCEEDED`` or ``INFEASIBLE``.
        client_id (str):
            Output only. The identifier of the client that originally
            requested this Trial. Each client is identified by a unique
            client_id. When a client asks for a suggestion, Vertex AI
            Vizier will assign it a Trial. The client should evaluate
            the Trial, complete it, and report back to Vertex AI Vizier.
            If suggestion is asked again by same client_id before the
            Trial is completed, the same Trial will be returned.
            Multiple clients with different client_ids can ask for
            suggestions simultaneously, each of them will get their own
            Trial.
        infeasible_reason (str):
            Output only. A human readable string describing why the
            Trial is infeasible. This is set only if Trial state is
            ``INFEASIBLE``.
        custom_job (str):
            Output only. The CustomJob name linked to the
            Trial. It's set for a HyperparameterTuningJob's
            Trial.
        web_access_uris (MutableMapping[str, str]):
            Output only. URIs for accessing `interactive
            shells <https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell>`__
            (one URI for each training node). Only available if this
            trial is part of a
            [HyperparameterTuningJob][google.cloud.aiplatform.v1.HyperparameterTuningJob]
            and the job's
            [trial_job_spec.enable_web_access][google.cloud.aiplatform.v1.CustomJobSpec.enable_web_access]
            field is ``true``.

            The keys are names of each node used for the trial; for
            example, ``workerpool0-0`` for the primary node,
            ``workerpool1-0`` for the first node in the second worker
            pool, and ``workerpool1-1`` for the second node in the
            second worker pool.

            The values are the URIs for each node's interactive shell.
    """
    class State(proto.Enum):
        r"""Describes a Trial state."""
        STATE_UNSPECIFIED = 0
        REQUESTED = 1
        ACTIVE = 2
        STOPPING = 3
        SUCCEEDED = 4
        INFEASIBLE = 5

    class Parameter(proto.Message):
        r"""A message representing a parameter to be tuned.

        Attributes:
            parameter_id (str):
                Output only. The ID of the parameter. The parameter should
                be defined in [StudySpec's
                Parameters][google.cloud.aiplatform.v1.StudySpec.parameters].
            value (google.protobuf.struct_pb2.Value):
                Output only. The value of the parameter. ``number_value``
                will be set if a parameter defined in StudySpec is in type
                'INTEGER', 'DOUBLE' or 'DISCRETE'. ``string_value`` will be
                set if a parameter defined in StudySpec is in type
                'CATEGORICAL'.
        """

        parameter_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        value: struct_pb2.Value = proto.Field(
            proto.MESSAGE,
            number=2,
            message=struct_pb2.Value,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    id: str = proto.Field(
        proto.STRING,
        number=2,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=3,
        enum=State,
    )
    parameters: MutableSequence[Parameter] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Parameter,
    )
    final_measurement: 'Measurement' = proto.Field(
        proto.MESSAGE,
        number=5,
        message='Measurement',
    )
    measurements: MutableSequence['Measurement'] = proto.RepeatedField(
        proto.MESSAGE,
        number=6,
        message='Measurement',
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    client_id: str = proto.Field(
        proto.STRING,
        number=9,
    )
    infeasible_reason: str = proto.Field(
        proto.STRING,
        number=10,
    )
    custom_job: str = proto.Field(
        proto.STRING,
        number=11,
    )
    web_access_uris: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=12,
    )


class StudySpec(proto.Message):
    r"""Represents specification of a Study.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        decay_curve_stopping_spec (google.cloud.aiplatform_v1.types.StudySpec.DecayCurveAutomatedStoppingSpec):
            The automated early stopping spec using decay
            curve rule.

            This field is a member of `oneof`_ ``automated_stopping_spec``.
        median_automated_stopping_spec (google.cloud.aiplatform_v1.types.StudySpec.MedianAutomatedStoppingSpec):
            The automated early stopping spec using
            median rule.

            This field is a member of `oneof`_ ``automated_stopping_spec``.
        convex_automated_stopping_spec (google.cloud.aiplatform_v1.types.StudySpec.ConvexAutomatedStoppingSpec):
            The automated early stopping spec using
            convex stopping rule.

            This field is a member of `oneof`_ ``automated_stopping_spec``.
        metrics (MutableSequence[google.cloud.aiplatform_v1.types.StudySpec.MetricSpec]):
            Required. Metric specs for the Study.
        parameters (MutableSequence[google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec]):
            Required. The set of parameters to tune.
        algorithm (google.cloud.aiplatform_v1.types.StudySpec.Algorithm):
            The search algorithm specified for the Study.
        observation_noise (google.cloud.aiplatform_v1.types.StudySpec.ObservationNoise):
            The observation noise level of the study.
            Currently only supported by the Vertex AI Vizier
            service. Not supported by
            HyperparameterTuningJob or TrainingPipeline.
        measurement_selection_type (google.cloud.aiplatform_v1.types.StudySpec.MeasurementSelectionType):
            Describe which measurement selection type
            will be used
    """
    class Algorithm(proto.Enum):
        r"""The available search algorithms for the Study."""
        ALGORITHM_UNSPECIFIED = 0
        GRID_SEARCH = 2
        RANDOM_SEARCH = 3

    class ObservationNoise(proto.Enum):
        r"""Describes the noise level of the repeated observations.
        "Noisy" means that the repeated observations with the same Trial
        parameters may lead to different metric evaluations.
        """
        OBSERVATION_NOISE_UNSPECIFIED = 0
        LOW = 1
        HIGH = 2

    class MeasurementSelectionType(proto.Enum):
        r"""This indicates which measurement to use if/when the service
        automatically selects the final measurement from previously reported
        intermediate measurements. Choose this based on two considerations:
        A) Do you expect your measurements to monotonically improve? If so,
        choose LAST_MEASUREMENT. On the other hand, if you're in a situation
        where your system can "over-train" and you expect the performance to
        get better for a while but then start declining, choose
        BEST_MEASUREMENT. B) Are your measurements significantly noisy
        and/or irreproducible? If so, BEST_MEASUREMENT will tend to be
        over-optimistic, and it may be better to choose LAST_MEASUREMENT. If
        both or neither of (A) and (B) apply, it doesn't matter which
        selection type is chosen.
        """
        MEASUREMENT_SELECTION_TYPE_UNSPECIFIED = 0
        LAST_MEASUREMENT = 1
        BEST_MEASUREMENT = 2

    class MetricSpec(proto.Message):
        r"""Represents a metric to optimize.

        Attributes:
            metric_id (str):
                Required. The ID of the metric. Must not
                contain whitespaces and must be unique amongst
                all MetricSpecs.
            goal (google.cloud.aiplatform_v1.types.StudySpec.MetricSpec.GoalType):
                Required. The optimization goal of the
                metric.
        """
        class GoalType(proto.Enum):
            r"""The available types of optimization goals."""
            GOAL_TYPE_UNSPECIFIED = 0
            MAXIMIZE = 1
            MINIMIZE = 2

        metric_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        goal: 'StudySpec.MetricSpec.GoalType' = proto.Field(
            proto.ENUM,
            number=2,
            enum='StudySpec.MetricSpec.GoalType',
        )

    class ParameterSpec(proto.Message):
        r"""Represents a single parameter to optimize.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            double_value_spec (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.DoubleValueSpec):
                The value spec for a 'DOUBLE' parameter.

                This field is a member of `oneof`_ ``parameter_value_spec``.
            integer_value_spec (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.IntegerValueSpec):
                The value spec for an 'INTEGER' parameter.

                This field is a member of `oneof`_ ``parameter_value_spec``.
            categorical_value_spec (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.CategoricalValueSpec):
                The value spec for a 'CATEGORICAL' parameter.

                This field is a member of `oneof`_ ``parameter_value_spec``.
            discrete_value_spec (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.DiscreteValueSpec):
                The value spec for a 'DISCRETE' parameter.

                This field is a member of `oneof`_ ``parameter_value_spec``.
            parameter_id (str):
                Required. The ID of the parameter. Must not
                contain whitespaces and must be unique amongst
                all ParameterSpecs.
            scale_type (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.ScaleType):
                How the parameter should be scaled. Leave unset for
                ``CATEGORICAL`` parameters.
            conditional_parameter_specs (MutableSequence[google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.ConditionalParameterSpec]):
                A conditional parameter node is active if the parameter's
                value matches the conditional node's parent_value_condition.

                If two items in conditional_parameter_specs have the same
                name, they must have disjoint parent_value_condition.
        """
        class ScaleType(proto.Enum):
            r"""The type of scaling that should be applied to this parameter."""
            SCALE_TYPE_UNSPECIFIED = 0
            UNIT_LINEAR_SCALE = 1
            UNIT_LOG_SCALE = 2
            UNIT_REVERSE_LOG_SCALE = 3

        class DoubleValueSpec(proto.Message):
            r"""Value specification for a parameter in ``DOUBLE`` type.

            Attributes:
                min_value (float):
                    Required. Inclusive minimum value of the
                    parameter.
                max_value (float):
                    Required. Inclusive maximum value of the
                    parameter.
                default_value (float):
                    A default value for a ``DOUBLE`` parameter that is assumed
                    to be a relatively good starting point. Unset value signals
                    that there is no offered starting point.

                    Currently only supported by the Vertex AI Vizier service.
                    Not supported by HyperparameterTuningJob or
                    TrainingPipeline.

                    This field is a member of `oneof`_ ``_default_value``.
            """

            min_value: float = proto.Field(
                proto.DOUBLE,
                number=1,
            )
            max_value: float = proto.Field(
                proto.DOUBLE,
                number=2,
            )
            default_value: float = proto.Field(
                proto.DOUBLE,
                number=4,
                optional=True,
            )

        class IntegerValueSpec(proto.Message):
            r"""Value specification for a parameter in ``INTEGER`` type.

            Attributes:
                min_value (int):
                    Required. Inclusive minimum value of the
                    parameter.
                max_value (int):
                    Required. Inclusive maximum value of the
                    parameter.
                default_value (int):
                    A default value for an ``INTEGER`` parameter that is assumed
                    to be a relatively good starting point. Unset value signals
                    that there is no offered starting point.

                    Currently only supported by the Vertex AI Vizier service.
                    Not supported by HyperparameterTuningJob or
                    TrainingPipeline.

                    This field is a member of `oneof`_ ``_default_value``.
            """

            min_value: int = proto.Field(
                proto.INT64,
                number=1,
            )
            max_value: int = proto.Field(
                proto.INT64,
                number=2,
            )
            default_value: int = proto.Field(
                proto.INT64,
                number=4,
                optional=True,
            )

        class CategoricalValueSpec(proto.Message):
            r"""Value specification for a parameter in ``CATEGORICAL`` type.

            Attributes:
                values (MutableSequence[str]):
                    Required. The list of possible categories.
                default_value (str):
                    A default value for a ``CATEGORICAL`` parameter that is
                    assumed to be a relatively good starting point. Unset value
                    signals that there is no offered starting point.

                    Currently only supported by the Vertex AI Vizier service.
                    Not supported by HyperparameterTuningJob or
                    TrainingPipeline.

                    This field is a member of `oneof`_ ``_default_value``.
            """

            values: MutableSequence[str] = proto.RepeatedField(
                proto.STRING,
                number=1,
            )
            default_value: str = proto.Field(
                proto.STRING,
                number=3,
                optional=True,
            )

        class DiscreteValueSpec(proto.Message):
            r"""Value specification for a parameter in ``DISCRETE`` type.

            Attributes:
                values (MutableSequence[float]):
                    Required. A list of possible values.
                    The list should be in increasing order and at
                    least 1e-10 apart. For instance, this parameter
                    might have possible settings of 1.5, 2.5, and
                    4.0. This list should not contain more than
                    1,000 values.
                default_value (float):
                    A default value for a ``DISCRETE`` parameter that is assumed
                    to be a relatively good starting point. Unset value signals
                    that there is no offered starting point. It automatically
                    rounds to the nearest feasible discrete point.

                    Currently only supported by the Vertex AI Vizier service.
                    Not supported by HyperparameterTuningJob or
                    TrainingPipeline.

                    This field is a member of `oneof`_ ``_default_value``.
            """

            values: MutableSequence[float] = proto.RepeatedField(
                proto.DOUBLE,
                number=1,
            )
            default_value: float = proto.Field(
                proto.DOUBLE,
                number=3,
                optional=True,
            )

        class ConditionalParameterSpec(proto.Message):
            r"""Represents a parameter spec with condition from its parent
            parameter.

            This message has `oneof`_ fields (mutually exclusive fields).
            For each oneof, at most one member field can be set at the same time.
            Setting any member of the oneof automatically clears all other
            members.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                parent_discrete_values (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.ConditionalParameterSpec.DiscreteValueCondition):
                    The spec for matching values from a parent parameter of
                    ``DISCRETE`` type.

                    This field is a member of `oneof`_ ``parent_value_condition``.
                parent_int_values (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.ConditionalParameterSpec.IntValueCondition):
                    The spec for matching values from a parent parameter of
                    ``INTEGER`` type.

                    This field is a member of `oneof`_ ``parent_value_condition``.
                parent_categorical_values (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec.ConditionalParameterSpec.CategoricalValueCondition):
                    The spec for matching values from a parent parameter of
                    ``CATEGORICAL`` type.

                    This field is a member of `oneof`_ ``parent_value_condition``.
                parameter_spec (google.cloud.aiplatform_v1.types.StudySpec.ParameterSpec):
                    Required. The spec for a conditional
                    parameter.
            """

            class DiscreteValueCondition(proto.Message):
                r"""Represents the spec to match discrete values from parent
                parameter.

                Attributes:
                    values (MutableSequence[float]):
                        Required. Matches values of the parent parameter of
                        'DISCRETE' type. All values must exist in
                        ``discrete_value_spec`` of parent parameter.

                        The Epsilon of the value matching is 1e-10.
                """

                values: MutableSequence[float] = proto.RepeatedField(
                    proto.DOUBLE,
                    number=1,
                )

            class IntValueCondition(proto.Message):
                r"""Represents the spec to match integer values from parent
                parameter.

                Attributes:
                    values (MutableSequence[int]):
                        Required. Matches values of the parent parameter of
                        'INTEGER' type. All values must lie in
                        ``integer_value_spec`` of parent parameter.
                """

                values: MutableSequence[int] = proto.RepeatedField(
                    proto.INT64,
                    number=1,
                )

            class CategoricalValueCondition(proto.Message):
                r"""Represents the spec to match categorical values from parent
                parameter.

                Attributes:
                    values (MutableSequence[str]):
                        Required. Matches values of the parent parameter of
                        'CATEGORICAL' type. All values must exist in
                        ``categorical_value_spec`` of parent parameter.
                """

                values: MutableSequence[str] = proto.RepeatedField(
                    proto.STRING,
                    number=1,
                )

            parent_discrete_values: 'StudySpec.ParameterSpec.ConditionalParameterSpec.DiscreteValueCondition' = proto.Field(
                proto.MESSAGE,
                number=2,
                oneof='parent_value_condition',
                message='StudySpec.ParameterSpec.ConditionalParameterSpec.DiscreteValueCondition',
            )
            parent_int_values: 'StudySpec.ParameterSpec.ConditionalParameterSpec.IntValueCondition' = proto.Field(
                proto.MESSAGE,
                number=3,
                oneof='parent_value_condition',
                message='StudySpec.ParameterSpec.ConditionalParameterSpec.IntValueCondition',
            )
            parent_categorical_values: 'StudySpec.ParameterSpec.ConditionalParameterSpec.CategoricalValueCondition' = proto.Field(
                proto.MESSAGE,
                number=4,
                oneof='parent_value_condition',
                message='StudySpec.ParameterSpec.ConditionalParameterSpec.CategoricalValueCondition',
            )
            parameter_spec: 'StudySpec.ParameterSpec' = proto.Field(
                proto.MESSAGE,
                number=1,
                message='StudySpec.ParameterSpec',
            )

        double_value_spec: 'StudySpec.ParameterSpec.DoubleValueSpec' = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof='parameter_value_spec',
            message='StudySpec.ParameterSpec.DoubleValueSpec',
        )
        integer_value_spec: 'StudySpec.ParameterSpec.IntegerValueSpec' = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof='parameter_value_spec',
            message='StudySpec.ParameterSpec.IntegerValueSpec',
        )
        categorical_value_spec: 'StudySpec.ParameterSpec.CategoricalValueSpec' = proto.Field(
            proto.MESSAGE,
            number=4,
            oneof='parameter_value_spec',
            message='StudySpec.ParameterSpec.CategoricalValueSpec',
        )
        discrete_value_spec: 'StudySpec.ParameterSpec.DiscreteValueSpec' = proto.Field(
            proto.MESSAGE,
            number=5,
            oneof='parameter_value_spec',
            message='StudySpec.ParameterSpec.DiscreteValueSpec',
        )
        parameter_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        scale_type: 'StudySpec.ParameterSpec.ScaleType' = proto.Field(
            proto.ENUM,
            number=6,
            enum='StudySpec.ParameterSpec.ScaleType',
        )
        conditional_parameter_specs: MutableSequence['StudySpec.ParameterSpec.ConditionalParameterSpec'] = proto.RepeatedField(
            proto.MESSAGE,
            number=10,
            message='StudySpec.ParameterSpec.ConditionalParameterSpec',
        )

    class DecayCurveAutomatedStoppingSpec(proto.Message):
        r"""The decay curve automated stopping rule builds a Gaussian
        Process Regressor to predict the final objective value of a
        Trial based on the already completed Trials and the intermediate
        measurements of the current Trial. Early stopping is requested
        for the current Trial if there is very low probability to exceed
        the optimal value found so far.

        Attributes:
            use_elapsed_duration (bool):
                True if
                [Measurement.elapsed_duration][google.cloud.aiplatform.v1.Measurement.elapsed_duration]
                is used as the x-axis of each Trials Decay Curve. Otherwise,
                [Measurement.step_count][google.cloud.aiplatform.v1.Measurement.step_count]
                will be used as the x-axis.
        """

        use_elapsed_duration: bool = proto.Field(
            proto.BOOL,
            number=1,
        )

    class MedianAutomatedStoppingSpec(proto.Message):
        r"""The median automated stopping rule stops a pending Trial if the
        Trial's best objective_value is strictly below the median
        'performance' of all completed Trials reported up to the Trial's
        last measurement. Currently, 'performance' refers to the running
        average of the objective values reported by the Trial in each
        measurement.

        Attributes:
            use_elapsed_duration (bool):
                True if median automated stopping rule applies on
                [Measurement.elapsed_duration][google.cloud.aiplatform.v1.Measurement.elapsed_duration].
                It means that elapsed_duration field of latest measurement
                of current Trial is used to compute median objective value
                for each completed Trials.
        """

        use_elapsed_duration: bool = proto.Field(
            proto.BOOL,
            number=1,
        )

    class ConvexAutomatedStoppingSpec(proto.Message):
        r"""Configuration for ConvexAutomatedStoppingSpec. When there are enough
        completed trials (configured by min_measurement_count), for pending
        trials with enough measurements and steps, the policy first computes
        an overestimate of the objective value at max_num_steps according to
        the slope of the incomplete objective value curve. No prediction can
        be made if the curve is completely flat. If the overestimation is
        worse than the best objective value of the completed trials, this
        pending trial will be early-stopped, but a last measurement will be
        added to the pending trial with max_num_steps and predicted
        objective value from the autoregression model.

        Attributes:
            max_step_count (int):
                Steps used in predicting the final objective for early
                stopped trials. In general, it's set to be the same as the
                defined steps in training / tuning. If not defined, it will
                learn it from the completed trials. When use_steps is false,
                this field is set to the maximum elapsed seconds.
            min_step_count (int):
                Minimum number of steps for a trial to complete. Trials
                which do not have a measurement with step_count >
                min_step_count won't be considered for early stopping. It's
                ok to set it to 0, and a trial can be early stopped at any
                stage. By default, min_step_count is set to be one-tenth of
                the max_step_count. When use_elapsed_duration is true, this
                field is set to the minimum elapsed seconds.
            min_measurement_count (int):
                The minimal number of measurements in a Trial.
                Early-stopping checks will not trigger if less than
                min_measurement_count+1 completed trials or pending trials
                with less than min_measurement_count measurements. If not
                defined, the default value is 5.
            learning_rate_parameter_name (str):
                The hyper-parameter name used in the tuning job that stands
                for learning rate. Leave it blank if learning rate is not in
                a parameter in tuning. The learning_rate is used to estimate
                the objective value of the ongoing trial.
            use_elapsed_duration (bool):
                This bool determines whether or not the rule is applied
                based on elapsed_secs or steps. If
                use_elapsed_duration==false, the early stopping decision is
                made according to the predicted objective values according
                to the target steps. If use_elapsed_duration==true,
                elapsed_secs is used instead of steps. Also, in this case,
                the parameters max_num_steps and min_num_steps are
                overloaded to contain max_elapsed_seconds and
                min_elapsed_seconds.
        """

        max_step_count: int = proto.Field(
            proto.INT64,
            number=1,
        )
        min_step_count: int = proto.Field(
            proto.INT64,
            number=2,
        )
        min_measurement_count: int = proto.Field(
            proto.INT64,
            number=3,
        )
        learning_rate_parameter_name: str = proto.Field(
            proto.STRING,
            number=4,
        )
        use_elapsed_duration: bool = proto.Field(
            proto.BOOL,
            number=5,
        )

    decay_curve_stopping_spec: DecayCurveAutomatedStoppingSpec = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof='automated_stopping_spec',
        message=DecayCurveAutomatedStoppingSpec,
    )
    median_automated_stopping_spec: MedianAutomatedStoppingSpec = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof='automated_stopping_spec',
        message=MedianAutomatedStoppingSpec,
    )
    convex_automated_stopping_spec: ConvexAutomatedStoppingSpec = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof='automated_stopping_spec',
        message=ConvexAutomatedStoppingSpec,
    )
    metrics: MutableSequence[MetricSpec] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=MetricSpec,
    )
    parameters: MutableSequence[ParameterSpec] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=ParameterSpec,
    )
    algorithm: Algorithm = proto.Field(
        proto.ENUM,
        number=3,
        enum=Algorithm,
    )
    observation_noise: ObservationNoise = proto.Field(
        proto.ENUM,
        number=6,
        enum=ObservationNoise,
    )
    measurement_selection_type: MeasurementSelectionType = proto.Field(
        proto.ENUM,
        number=7,
        enum=MeasurementSelectionType,
    )


class Measurement(proto.Message):
    r"""A message representing a Measurement of a Trial. A
    Measurement contains the Metrics got by executing a Trial using
    suggested hyperparameter values.

    Attributes:
        elapsed_duration (google.protobuf.duration_pb2.Duration):
            Output only. Time that the Trial has been
            running at the point of this Measurement.
        step_count (int):
            Output only. The number of steps the machine
            learning model has been trained for. Must be
            non-negative.
        metrics (MutableSequence[google.cloud.aiplatform_v1.types.Measurement.Metric]):
            Output only. A list of metrics got by
            evaluating the objective functions using
            suggested Parameter values.
    """

    class Metric(proto.Message):
        r"""A message representing a metric in the measurement.

        Attributes:
            metric_id (str):
                Output only. The ID of the Metric. The Metric should be
                defined in [StudySpec's
                Metrics][google.cloud.aiplatform.v1.StudySpec.metrics].
            value (float):
                Output only. The value for this metric.
        """

        metric_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        value: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )

    elapsed_duration: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )
    step_count: int = proto.Field(
        proto.INT64,
        number=2,
    )
    metrics: MutableSequence[Metric] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=Metric,
    )


__all__ = tuple(sorted(__protobuf__.manifest))

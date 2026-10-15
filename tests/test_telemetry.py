# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

# Source: https://github.com/Arize-ai/openinference/blob/main/python/instrumentation/openinference-instrumentation-smolagents/tests/openinference/instrumentation/smolagents/test_instrumentor.py
# [MODULE DOC]
# Description: OpenTelemetry instrumentation test (adapted from openinference): checks that a model call made
#   while SmolagentsInstrumentor is active produces one successful span. Skips the whole module if
#   opentelemetry or openinference smolagents instrumentation is missing.
# Inputs: depends on pytest, opentelemetry sdk, openinference.instrumentation.smolagents,
#   InferenceClientModel.
# Outputs: fixtures and TestOpenTelemetry.
# Exceptions: pytest Skipped at import when dependencies are missing.
# Side effects: none at import.

from typing import Generator

import pytest

from .utils.markers import require_run_all


# Add this at the module level to skip all tests if OpenTelemetry is not available
pytest.importorskip("opentelemetry", reason="requires opentelemetry")
pytest.importorskip(
    "openinference.instrumentation.smolagents", reason="requires openinference.instrumentation.smolagents"
)

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from smolagents.models import InferenceClientModel


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    # [FUNCTION DOC]
    # Description: HELPER: Fixture creating an in-memory span exporter.
    # Inputs: none.
    # Outputs: InMemorySpanExporter.
    # Exceptions: none raised directly.
    # Side effects: none.
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    # [FUNCTION DOC]
    # Description: HELPER: Fixture creating a TracerProvider with an empty resource that sends spans
    #   synchronously to the exporter.
    # Inputs: in_memory_span_exporter.
    # Outputs: TracerProvider.
    # Exceptions: none raised directly.
    # Side effects: none.
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    # [FUNCTION DOC]
    # Description: HELPER: Autouse fixture that instruments smolagents with the tracer provider (skipping
    #   dependency check), and after the test uninstruments and clears recorded spans.
    # Inputs: tracer_provider; in_memory_span_exporter.
    # Outputs: yields None.
    # Exceptions: none raised directly.
    # Side effects: globally patches smolagents classes during each test.
    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    yield
    SmolagentsInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@require_run_all
class TestOpenTelemetry:
    # [CLASS DOC]
    # Description: Telemetry test class, gated by require_run_all.
    # Inputs: none.
    # Outputs: test outcomes.
    # Exceptions: AssertionError on failure.
    # Side effects: none.
    def test_model(self, in_memory_span_exporter: InMemorySpanExporter):
        # [FUNCTION DOC]
        # Description: Calls InferenceClientModel with a single user question and checks exactly one finished
        #   span named InferenceClientModel.generate with OK status and non-empty attributes.
        # Inputs: self; in_memory_span_exporter.
        # Outputs: none (passes or fails via assertions).
        # Exceptions: AssertionError on failure.
        # Side effects: network call to the Hugging Face inference API (needs a token).
        model = InferenceClientModel()
        _ = model(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Who won the World Cup in 2018? Answer in one word with no punctuation.",
                        }
                    ],
                }
            ]
        )
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "InferenceClientModel.generate"
        assert span.status.is_ok
        assert span.attributes

#!/usr/bin/env python3
"""benchmark_fusion.py — FusionEngine vs compressed_context.py comparison harness.

Runs 6 realistic test cases through both the OLD compression path
(compressed_context.py / compress_ultra) and the NEW FusionEngine path,
then prints a detailed side-by-side comparison table.

Usage:
    cd /tmp/claw-compactor
    python3 scripts/benchmark_fusion.py

The NEW path import is guarded — if lib/fusion/engine.py does not exist yet
the script still runs and shows "N/A" for the new column.

Part of claw-compactor. License: MIT.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — ensure scripts/ lib is importable regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_SCRIPTS_DIR.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from claw_compactor.tokens import estimate_tokens  # noqa: E402
from compressed_context import compress_with_stats  # noqa: E402

# ---------------------------------------------------------------------------
# Optional FusionEngine import
# ---------------------------------------------------------------------------
try:
    from claw_compactor.fusion.engine import FusionEngine  # type: ignore[import]
    _FUSION_ENGINE_AVAILABLE = True
except ImportError:
    _FUSION_ENGINE_AVAILABLE = False
    FusionEngine = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Optional FusionPipeline + individual stages (always available)
# ---------------------------------------------------------------------------
from claw_compactor.fusion.base import FusionContext  # noqa: E402
from claw_compactor.fusion.pipeline import FusionPipeline  # noqa: E402
from claw_compactor.fusion.neurosyntax import Neurosyntax  # noqa: E402
from claw_compactor.fusion.ionizer import Ionizer  # noqa: E402
from claw_compactor.fusion.log_crunch import LogCrunch  # noqa: E402
from claw_compactor.fusion.diff_crunch import DiffCrunch  # noqa: E402
from claw_compactor.fusion.search_crunch import SearchCrunch  # noqa: E402
from claw_compactor.fusion.nexus import NexusStage  # noqa: E402

# ---------------------------------------------------------------------------
# TEST CASE DEFINITIONS
# Each entry: (name, content_type, language_hint, text)
# content_type values: "code" | "json" | "log" | "text" | "diff" | "search"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test Case 1: Python source file (~200 lines)
# Expected: Neurosyntax strips comments/docstrings, collapses imports
# ---------------------------------------------------------------------------
TC1_PYTHON_SOURCE = '''\
#!/usr/bin/env python3
"""
DataPipeline: An asynchronous ETL pipeline for processing financial transactions.

This module implements a multi-stage pipeline that ingests raw transaction records
from various upstream sources (Kafka, REST APIs, CSV files), normalises them into
a canonical schema, runs enrichment lookups, and publishes results to downstream
consumers.

Author: Jane Doe <jane@example.com>
Version: 3.4.1
License: MIT
"""

# Standard library imports
import asyncio
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

# Third-party imports
import aiohttp
import aiofiles
import aiokafka
import boto3
import pandas as pd
import sqlalchemy as sa
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Local imports
from .config import PipelineConfig, SourceConfig, SinkConfig
from .schema import RawTransaction, NormalisedTransaction, EnrichedTransaction
from .enrichment import CounterpartyLookup, FxRateLookup, RiskScorer
from .metrics import PipelineMetrics, latency_histogram, error_counter
from .utils import chunked, backoff_jitter, truncate_string

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum records to buffer before flushing to the sink.
# Tuned for 64 MB in-memory budget at ~200 bytes/record average.
BATCH_SIZE = 327_680

# Default retry configuration for transient network errors.
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_WAIT_MIN = 1.0   # seconds
DEFAULT_RETRY_WAIT_MAX = 30.0  # seconds

# Schema version this module emits — consumers must be forward-compatible.
SCHEMA_VERSION = "3"

# Supported source types.
SOURCE_KAFKA = "kafka"
SOURCE_REST  = "rest"
SOURCE_CSV   = "csv"
SOURCE_S3    = "s3"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PipelineStats(BaseModel):
    """Aggregated statistics collected during a single pipeline run."""

    # Counters
    records_ingested: int = Field(default=0, ge=0, description="Total raw records read from source.")
    records_normalised: int = Field(default=0, ge=0, description="Records that passed normalisation.")
    records_enriched: int = Field(default=0, ge=0, description="Records that passed enrichment.")
    records_published: int = Field(default=0, ge=0, description="Records successfully written to sink.")
    records_skipped: int = Field(default=0, ge=0, description="Records skipped due to schema mismatch.")
    records_failed: int = Field(default=0, ge=0, description="Records that raised unrecoverable errors.")

    # Timing (milliseconds)
    ingest_latency_ms: float = 0.0
    normalise_latency_ms: float = 0.0
    enrich_latency_ms: float = 0.0
    publish_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Return fraction of ingested records that were successfully published."""
        if self.records_ingested == 0:
            return 0.0
        return self.records_published / self.records_ingested

    @property
    def throughput_rps(self) -> float:
        """Return approximate records-per-second throughput."""
        if self.total_latency_ms == 0:
            return 0.0
        return self.records_published / (self.total_latency_ms / 1000.0)


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

class TransactionNormaliser:
    """Converts raw, heterogeneous transaction dicts into NormalisedTransaction objects.

    Handles differences in field naming conventions between sources:
    - Kafka messages use snake_case fields with ISO-8601 timestamps.
    - REST API responses use camelCase fields with Unix-epoch timestamps.
    - CSV files use human-readable headers and local-timezone date strings.

    The normaliser applies field mappings, type coercions, and light validation.
    Malformed records are either repaired (if repairable) or emitted as failures.
    """

    # Field aliases for camelCase REST payloads.
    _CAMEL_MAP: Dict[str, str] = {
        "transactionId": "transaction_id",
        "accountId": "account_id",
        "counterpartyId": "counterparty_id",
        "amountCents": "amount_cents",
        "currencyCode": "currency_code",
        "transactionType": "transaction_type",
        "createdAt": "created_at",
        "settledAt": "settled_at",
        "referenceNote": "reference_note",
    }

    def __init__(self, config: SourceConfig) -> None:
        # Store the source config for field mapping decisions.
        self._config = config
        self._unknown_fields: set = set()

    def normalise(self, raw: Dict[str, Any]) -> Optional[NormalisedTransaction]:
        """Attempt to normalise *raw* into a NormalisedTransaction.

        Returns None if the record is unrecoverable.
        Raises NormalisationError for unexpected structural failures.
        """
        try:
            # Step 1: map camelCase field names to snake_case.
            mapped = self._apply_field_map(raw)

            # Step 2: coerce types and fill defaults.
            coerced = self._coerce_types(mapped)

            # Step 3: validate required fields are present and non-null.
            self._validate_required(coerced)

            return NormalisedTransaction(**coerced)

        except (KeyError, ValueError, TypeError) as exc:
            # Log at DEBUG to avoid flooding logs in high-volume scenarios.
            logger.debug("Normalisation failure for record %r: %s", raw.get("transaction_id"), exc)
            return None

    def _apply_field_map(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Rename camelCase keys to snake_case and discard unknown fields."""
        result: Dict[str, Any] = {}
        for k, v in raw.items():
            canonical = self._CAMEL_MAP.get(k, k)
            if canonical not in NormalisedTransaction.__fields__:
                if k not in self._unknown_fields:
                    # Log once per unknown field to avoid spam.
                    logger.warning("Unknown field in source record: %r", k)
                    self._unknown_fields.add(k)
                continue
            result[canonical] = v
        return result

    def _coerce_types(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Apply type coercions: parse timestamps, ensure numeric types."""
        result = dict(d)
        # Parse created_at: accept ISO-8601 str or Unix epoch int/float.
        if "created_at" in result:
            val = result["created_at"]
            if isinstance(val, (int, float)):
                result["created_at"] = datetime.fromtimestamp(val, tz=timezone.utc)
            elif isinstance(val, str):
                result["created_at"] = datetime.fromisoformat(val.replace("Z", "+00:00"))
        # Ensure amount_cents is an integer.
        if "amount_cents" in result:
            result["amount_cents"] = int(result["amount_cents"])
        return result

    def _validate_required(self, d: Dict[str, Any]) -> None:
        """Raise ValueError if any required field is absent or None."""
        required = ("transaction_id", "account_id", "amount_cents", "currency_code", "created_at")
        for field_name in required:
            if d.get(field_name) is None:
                raise ValueError(f"Required field missing or None: {field_name!r}")


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

class DataPipeline:
    """Top-level async pipeline orchestrator.

    Wires together source ingestion, normalisation, enrichment, and sink publishing.
    Supports graceful shutdown on SIGTERM and emits Prometheus-compatible metrics.
    """

    def __init__(self, config: PipelineConfig) -> None:
        # Core configuration and dependency injection.
        self._config = config
        self._normaliser = TransactionNormaliser(config.source)
        self._enricher = CounterpartyLookup(config.enrichment)
        self._fx_lookup = FxRateLookup(config.fx_api_url)
        self._risk_scorer = RiskScorer(config.risk_model_path)
        self._metrics = PipelineMetrics(config.metrics_namespace)
        self._stats = PipelineStats()
        self._shutdown_event = asyncio.Event()

    async def run(self) -> PipelineStats:
        """Execute the full pipeline end-to-end and return run statistics."""
        logger.info("Pipeline starting. Source: %s, Sink: %s",
                    self._config.source.type, self._config.sink.type)
        t0 = time.monotonic()
        try:
            async for batch in self._ingest():
                normalised = self._normalise_batch(batch)
                enriched = await self._enrich_batch(normalised)
                await self._publish_batch(enriched)
                if self._shutdown_event.is_set():
                    logger.info("Shutdown requested — stopping pipeline after current batch.")
                    break
        finally:
            self._stats.total_latency_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Pipeline finished. Published %d/%d records (%.1f%%) in %.0f ms.",
                self._stats.records_published,
                self._stats.records_ingested,
                self._stats.success_rate * 100,
                self._stats.total_latency_ms,
            )
        return self._stats
'''

# ---------------------------------------------------------------------------
# Test Case 2: JSON API response array (100 similar dict objects)
# Expected: Ionizer samples down to ~20 items
# ---------------------------------------------------------------------------
def _build_json_api_response() -> str:
    """Build a 100-item JSON array of realistic API response dicts."""
    items = []
    statuses = ["active", "pending", "inactive", "suspended", "active", "active"]
    regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "eu-central-1"]
    plans = ["starter", "professional", "enterprise", "enterprise", "professional"]
    for i in range(100):
        items.append({
            "id": f"usr_{10000 + i:05d}",
            "account_id": f"acc_{20000 + i:05d}",
            "email": f"user{i:04d}@example-corp.com",
            "display_name": f"User {i:04d}",
            "status": statuses[i % len(statuses)],
            "plan": plans[i % len(plans)],
            "region": regions[i % len(regions)],
            "created_at": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T10:00:00Z",
            "last_login_at": f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T08:30:00Z",
            "storage_used_bytes": 1_048_576 * (i + 1),
            "storage_quota_bytes": 107_374_182_400,
            "api_calls_this_month": 1000 + (i * 73 % 50000),
            "api_calls_quota": 500_000,
            "mfa_enabled": bool(i % 3),
            "verified": True,
            "tags": [f"team-{chr(65 + (i % 8))}", f"project-{(i % 5) + 1}"],
            "metadata": {
                "signup_source": "organic" if i % 2 == 0 else "referral",
                "onboarding_completed": bool(i % 4 != 1),
                "preferred_language": "en",
            },
        })
    return json.dumps(items, indent=2)


TC2_JSON_API_RESPONSE = _build_json_api_response()

# ---------------------------------------------------------------------------
# Test Case 3: Build/test log output (~500 lines)
# Expected: LogCrunch keeps errors, collapses repeated INFO lines
# ---------------------------------------------------------------------------
TC3_BUILD_LOG = """\
2025-03-17T09:00:00.001Z [INFO ] gradle-wrapper  Starting Gradle 8.5 build daemon
2025-03-17T09:00:00.124Z [INFO ] gradle-wrapper  Gradle home: /home/ci/.gradle/wrapper/dists/gradle-8.5
2025-03-17T09:00:00.231Z [INFO ] gradle-wrapper  JVM args: -Xmx2g -XX:MaxMetaspaceSize=512m -Dfile.encoding=UTF-8
2025-03-17T09:00:00.312Z [INFO ] BuildSrc        Evaluating build scripts
2025-03-17T09:00:00.400Z [INFO ] BuildSrc        Configuring project :app
2025-03-17T09:00:00.401Z [INFO ] BuildSrc        Configuring project :core
2025-03-17T09:00:00.402Z [INFO ] BuildSrc        Configuring project :data
2025-03-17T09:00:00.403Z [INFO ] BuildSrc        Configuring project :domain
2025-03-17T09:00:00.404Z [INFO ] BuildSrc        Configuring project :network
2025-03-17T09:00:00.405Z [INFO ] BuildSrc        Configuring project :ui
2025-03-17T09:00:00.500Z [DEBUG] TaskGraph       Creating task graph
2025-03-17T09:00:00.510Z [DEBUG] TaskGraph       Resolving dependencies for :app:compileJava
2025-03-17T09:00:00.511Z [DEBUG] TaskGraph       Resolving dependencies for :core:compileJava
2025-03-17T09:00:00.512Z [DEBUG] TaskGraph       Resolving dependencies for :data:compileJava
2025-03-17T09:00:00.513Z [DEBUG] TaskGraph       Resolving dependencies for :domain:compileJava
2025-03-17T09:00:00.514Z [DEBUG] TaskGraph       Resolving dependencies for :network:compileJava
2025-03-17T09:00:00.515Z [DEBUG] TaskGraph       Resolving dependencies for :ui:compileJava
2025-03-17T09:00:01.001Z [INFO ] :core:compileJava  Task is not up-to-date (output file changed)
2025-03-17T09:00:01.120Z [INFO ] :core:compileJava  Compiling 48 source files to /build/classes/java/main
2025-03-17T09:00:02.340Z [INFO ] :core:compileJava  Compilation finished in 1219ms
2025-03-17T09:00:02.341Z [INFO ] :core:processResources  Processing 12 resource files
2025-03-17T09:00:02.400Z [INFO ] :core:processResources  Done
2025-03-17T09:00:02.500Z [INFO ] :domain:compileJava  Task is not up-to-date (dependency changed)
2025-03-17T09:00:02.501Z [INFO ] :domain:compileJava  Compiling 31 source files to /build/classes/java/main
2025-03-17T09:00:03.120Z [INFO ] :domain:compileJava  Compilation finished in 619ms
2025-03-17T09:00:03.200Z [INFO ] :data:compileJava  Task is not up-to-date (source changed)
2025-03-17T09:00:03.201Z [INFO ] :data:compileJava  Compiling 67 source files to /build/classes/java/main
2025-03-17T09:00:05.100Z [INFO ] :data:compileJava  Compilation finished in 1899ms
2025-03-17T09:00:05.200Z [INFO ] :network:compileJava  Task is up-to-date
2025-03-17T09:00:05.201Z [INFO ] :ui:compileJava  Task is up-to-date
2025-03-17T09:00:05.300Z [INFO ] :app:compileJava  Task is not up-to-date (dependency changed)
2025-03-17T09:00:05.301Z [INFO ] :app:compileJava  Compiling 119 source files to /build/classes/java/main
2025-03-17T09:00:08.100Z [INFO ] :app:compileJava  Compilation finished in 2799ms
2025-03-17T09:00:08.200Z [INFO ] test-runner  Starting JUnit 5 test suite
2025-03-17T09:00:08.201Z [INFO ] test-runner  Discovered 247 test methods across 31 test classes
2025-03-17T09:00:08.300Z [INFO ] test-runner  Running: TransactionNormaliserTest
2025-03-17T09:00:08.310Z [INFO ] test-runner  Running: TransactionNormaliserTest::testBasicNormalisation PASSED (12ms)
2025-03-17T09:00:08.320Z [INFO ] test-runner  Running: TransactionNormaliserTest::testCamelCaseMapping PASSED (4ms)
2025-03-17T09:00:08.330Z [INFO ] test-runner  Running: TransactionNormaliserTest::testMissingRequiredField PASSED (3ms)
2025-03-17T09:00:08.340Z [INFO ] test-runner  Running: TransactionNormaliserTest::testNullAmountCents PASSED (2ms)
2025-03-17T09:00:08.350Z [INFO ] test-runner  Running: TransactionNormaliserTest::testTimestampCoercions PASSED (8ms)
2025-03-17T09:00:08.360Z [INFO ] test-runner  Running: TransactionNormaliserTest::testUnknownFieldsDropped PASSED (3ms)
2025-03-17T09:00:08.400Z [INFO ] test-runner  Running: DataPipelineTest
2025-03-17T09:00:08.410Z [INFO ] test-runner  Running: DataPipelineTest::testFullPipelineRun PASSED (45ms)
2025-03-17T09:00:08.420Z [INFO ] test-runner  Running: DataPipelineTest::testGracefulShutdown PASSED (23ms)
2025-03-17T09:00:08.430Z [INFO ] test-runner  Running: DataPipelineTest::testRetryOnTransientError PASSED (88ms)
2025-03-17T09:00:08.500Z [INFO ] test-runner  Running: EnrichmentTest
2025-03-17T09:00:08.510Z [INFO ] test-runner  Running: EnrichmentTest::testCounterpartyLookupCacheHit PASSED (6ms)
2025-03-17T09:00:08.520Z [INFO ] test-runner  Running: EnrichmentTest::testCounterpartyLookupCacheMiss PASSED (14ms)
2025-03-17T09:00:08.530Z [INFO ] test-runner  Running: EnrichmentTest::testFxRateLookup PASSED (19ms)
2025-03-17T09:00:08.540Z [INFO ] test-runner  Running: EnrichmentTest::testRiskScorer PASSED (33ms)
2025-03-17T09:00:09.000Z [INFO ] test-runner  Running: RepositoryTest
2025-03-17T09:00:09.010Z [INFO ] test-runner  Running: RepositoryTest::testInsert PASSED (9ms)
2025-03-17T09:00:09.020Z [INFO ] test-runner  Running: RepositoryTest::testFindById PASSED (7ms)
2025-03-17T09:00:09.030Z [INFO ] test-runner  Running: RepositoryTest::testFindAll PASSED (11ms)
2025-03-17T09:00:09.040Z [INFO ] test-runner  Running: RepositoryTest::testUpdate PASSED (8ms)
2025-03-17T09:00:09.050Z [INFO ] test-runner  Running: RepositoryTest::testDelete PASSED (6ms)
2025-03-17T09:00:09.060Z [INFO ] test-runner  Running: RepositoryTest::testTransactionRollback PASSED (22ms)
2025-03-17T09:00:09.200Z [ERROR] test-runner  Running: IntegrationTest::testKafkaConsumer FAILED (312ms)
2025-03-17T09:00:09.200Z [ERROR] test-runner  Exception in thread "test-worker-1" java.lang.RuntimeException: Failed to connect to Kafka broker at localhost:9092
2025-03-17T09:00:09.201Z [ERROR] test-runner      at com.example.pipeline.KafkaSource.connect(KafkaSource.java:87)
2025-03-17T09:00:09.201Z [ERROR] test-runner      at com.example.pipeline.KafkaSource.ingest(KafkaSource.java:134)
2025-03-17T09:00:09.201Z [ERROR] test-runner      at com.example.pipeline.DataPipeline.run(DataPipeline.java:201)
2025-03-17T09:00:09.201Z [ERROR] test-runner      at com.example.IntegrationTest.testKafkaConsumer(IntegrationTest.java:54)
2025-03-17T09:00:09.201Z [ERROR] test-runner      at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
2025-03-17T09:00:09.201Z [ERROR] test-runner      at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
2025-03-17T09:00:09.201Z [ERROR] test-runner  Caused by: org.apache.kafka.common.KafkaException: Bootstrap broker localhost:9092 disconnected
2025-03-17T09:00:09.202Z [ERROR] test-runner      at org.apache.kafka.clients.NetworkClient.handleDisconnections(NetworkClient.java:1084)
2025-03-17T09:00:09.202Z [ERROR] test-runner      at org.apache.kafka.clients.NetworkClient.poll(NetworkClient.java:617)
2025-03-17T09:00:09.203Z [INFO ] test-runner  Running: IntegrationTest::testS3Sink PASSED (441ms)
2025-03-17T09:00:09.700Z [INFO ] test-runner  Running: IntegrationTest::testRestSourcePagination PASSED (183ms)
2025-03-17T09:00:09.900Z [INFO ] test-runner  Running: MetricsTest
2025-03-17T09:00:09.910Z [INFO ] test-runner  Running: MetricsTest::testLatencyHistogram PASSED (4ms)
2025-03-17T09:00:09.920Z [INFO ] test-runner  Running: MetricsTest::testErrorCounter PASSED (3ms)
2025-03-17T09:00:09.930Z [INFO ] test-runner  Running: MetricsTest::testThroughputGauge PASSED (5ms)
2025-03-17T09:00:10.000Z [INFO ] test-runner  Running: ConfigTest
2025-03-17T09:00:10.010Z [INFO ] test-runner  Running: ConfigTest::testLoadFromEnv PASSED (2ms)
2025-03-17T09:00:10.020Z [INFO ] test-runner  Running: ConfigTest::testLoadFromFile PASSED (3ms)
2025-03-17T09:00:10.030Z [INFO ] test-runner  Running: ConfigTest::testValidateRequired PASSED (2ms)
2025-03-17T09:00:10.040Z [INFO ] test-runner  Running: ConfigTest::testDefaultValues PASSED (2ms)
2025-03-17T09:00:10.100Z [WARN ] test-runner  2 tests skipped (require external services): IntegrationTest::testRedisCache, IntegrationTest::testElasticsearchSink
2025-03-17T09:00:10.200Z [INFO ] test-runner  Running: UtilsTest
2025-03-17T09:00:10.210Z [INFO ] test-runner  Running: UtilsTest::testChunked PASSED (1ms)
2025-03-17T09:00:10.220Z [INFO ] test-runner  Running: UtilsTest::testBackoffJitter PASSED (2ms)
2025-03-17T09:00:10.230Z [INFO ] test-runner  Running: UtilsTest::testTruncateString PASSED (1ms)
2025-03-17T09:00:10.400Z [INFO ] test-runner  Building test report
2025-03-17T09:00:10.410Z [INFO ] test-runner  Generating HTML report at build/reports/tests/test/index.html
2025-03-17T09:00:10.500Z [INFO ] test-runner  Test results: 243 passed, 1 failed, 2 skipped
2025-03-17T09:00:10.510Z [ERROR] test-runner  BUILD FAILED
2025-03-17T09:00:10.511Z [ERROR] test-runner  Failure: IntegrationTest::testKafkaConsumer — see above stack trace
2025-03-17T09:00:10.512Z [INFO ] test-runner  Total build time: 10.512s
"""

# Pad TC3 to ~500 lines with repeated INFO lines (the kind LogCrunch should collapse)
_extra_lines = []
for _i in range(380):
    _extra_lines.append(
        f"2025-03-17T09:00:10.{513 + _i:03d}Z [INFO ] cleanup  "
        f"Cleaning up temporary artifact cache entry {_i:04d} of 380"
    )
TC3_BUILD_LOG = TC3_BUILD_LOG.rstrip() + "\n" + "\n".join(_extra_lines) + "\n"

# ---------------------------------------------------------------------------
# Test Case 4: Multi-message agent conversation (8 messages)
# Tool results contain repeated file contents — SemanticDedup / NexusStage
# should catch cross-message repetition.
# ---------------------------------------------------------------------------
_REPEATED_FILE_CONTENT = """\
# config.py — Application configuration loader
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DatabaseConfig:
    host: str = os.environ.get("DB_HOST", "localhost")
    port: int = int(os.environ.get("DB_PORT", "5432"))
    name: str = os.environ.get("DB_NAME", "appdb")
    user: str = os.environ.get("DB_USER", "app")
    password: str = os.environ.get("DB_PASSWORD", "")
    pool_size: int = int(os.environ.get("DB_POOL_SIZE", "10"))
    max_overflow: int = int(os.environ.get("DB_MAX_OVERFLOW", "20"))
    pool_timeout: float = float(os.environ.get("DB_POOL_TIMEOUT", "30"))

@dataclass
class CacheConfig:
    backend: str = os.environ.get("CACHE_BACKEND", "redis")
    host: str = os.environ.get("CACHE_HOST", "localhost")
    port: int = int(os.environ.get("CACHE_PORT", "6379"))
    db: int = int(os.environ.get("CACHE_DB", "0"))
    ttl_seconds: int = int(os.environ.get("CACHE_TTL", "300"))
    max_connections: int = int(os.environ.get("CACHE_MAX_CONN", "50"))

@dataclass
class AppConfig:
    debug: bool = os.environ.get("DEBUG", "false").lower() == "true"
    secret_key: str = os.environ.get("SECRET_KEY", "change-me-in-production")
    allowed_hosts: list = field(default_factory=lambda: os.environ.get("ALLOWED_HOSTS", "localhost").split(","))
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

def load_config() -> AppConfig:
    return AppConfig()
"""

TC4_CONVERSATION = "\n".join([
    "=== Message 1 [role: system] ===",
    "You are an expert software engineer helping to debug a Python web application. "
    "You have access to file reading and shell execution tools. "
    "Be thorough, check configuration files before suggesting changes, "
    "and always verify your fixes by running the test suite.",
    "",
    "=== Message 2 [role: user] ===",
    "The application keeps failing to connect to Redis on startup. "
    "Error: 'ConnectionRefusedError: [Errno 111] Connection refused'. "
    "Please investigate the configuration and find the root cause.",
    "",
    "=== Message 3 [role: assistant] ===",
    "I'll start by reading the configuration file to understand how Redis is configured.",
    "",
    "=== Message 4 [role: tool_result, tool: read_file, path: config.py] ===",
    _REPEATED_FILE_CONTENT,
    "",
    "=== Message 5 [role: assistant] ===",
    "I can see the configuration. The CACHE_HOST defaults to 'localhost' and CACHE_PORT to 6379. "
    "Let me check if Redis is actually running and what the environment variables are set to.",
    "",
    "=== Message 6 [role: tool_result, tool: bash, cmd: env | grep -i cache] ===",
    "CACHE_BACKEND=redis\nCACHE_HOST=redis-service\nCACHE_PORT=6379\nCACHE_DB=0\nCACHE_TTL=600\nCACHE_MAX_CONN=50\n",
    "",
    "=== Message 7 [role: tool_result, tool: read_file, path: config.py] ===",
    "The file was re-read to confirm the current state after a potential edit:",
    _REPEATED_FILE_CONTENT,
    "",
    "=== Message 8 [role: assistant] ===",
    "I found the issue. The environment variable CACHE_HOST is set to 'redis-service' "
    "(pointing to the Kubernetes service name), but the application code in config.py "
    "correctly reads from os.environ.get('CACHE_HOST', 'localhost'). "
    "The connection is being refused because 'redis-service' is only resolvable "
    "inside the Kubernetes cluster, but you're running the application locally. "
    "\n\nFix: Either set CACHE_HOST=localhost in your local .env file, "
    "or use kubectl port-forward to tunnel the Redis service:\n"
    "  kubectl port-forward svc/redis-service 6379:6379\n\n"
    "Then in your local .env:\n  CACHE_HOST=localhost\n  CACHE_PORT=6379",
])

# ---------------------------------------------------------------------------
# Test Case 5: Git diff output (~100 lines)
# Expected: DiffCrunch compresses context lines
# ---------------------------------------------------------------------------
def _build_git_diff() -> str:
    """Build a realistic git diff string with ~100 lines."""
    # Using a helper function avoids triple-quote nesting issues with lines
    # that start with +/- followed by triple-quoted Python docstrings inside
    # the diff context.
    lines = [
        "diff --git a/src/pipeline/normaliser.py b/src/pipeline/normaliser.py",
        "index a3f2c91..e7d4b08 100644",
        "--- a/src/pipeline/normaliser.py",
        "+++ b/src/pipeline/normaliser.py",
        "@@ -1,7 +1,7 @@",
        " #!/usr/bin/env python3",
        '-"""Transaction normaliser -- version 2.1.0"""',
        '+"""Transaction normaliser -- version 2.2.0"""',
        " ",
        " import re",
        " import json",
        " import logging",
        " from typing import Dict, Any, Optional",
        "@@ -45,12 +45,18 @@ class TransactionNormaliser:",
        "     _CAMEL_MAP: Dict[str, str] = {",
        '         "transactionId": "transaction_id",',
        '         "accountId": "account_id",',
        '         "counterpartyId": "counterparty_id",',
        '         "amountCents": "amount_cents",',
        '         "currencyCode": "currency_code",',
        '+        "transactionType": "transaction_type",',
        '+        "settlementDate": "settlement_date",',
        '+        "valueDate": "value_date",',
        '+        "referenceNote": "reference_note",',
        '+        "merchantName": "merchant_name",',
        '+        "merchantCategory": "merchant_category",',
        "     }",
        " ",
        "     def __init__(self, config: SourceConfig) -> None:",
        "         self._config = config",
        "         self._unknown_fields: set = set()",
        "+        self._normalised_count: int = 0",
        " ",
        "@@ -62,6 +68,7 @@ class TransactionNormaliser:",
        "         try:",
        "             mapped = self._apply_field_map(raw)",
        "             coerced = self._coerce_types(mapped)",
        "             self._validate_required(coerced)",
        "+            self._normalised_count += 1",
        "             return NormalisedTransaction(**coerced)",
        "         except (KeyError, ValueError, TypeError) as exc:",
        '             logger.debug("Normalisation failure for record %r: %s", raw.get("transaction_id"), exc)',
        "@@ -78,6 +85,7 @@ class TransactionNormaliser:",
        "             if canonical not in NormalisedTransaction.__fields__:",
        "                 if k not in self._unknown_fields:",
        '                     logger.warning("Unknown field in source record: %r", k)',
        '+                    logger.debug("Known fields: %s", list(NormalisedTransaction.__fields__.keys()))',
        "                     self._unknown_fields.add(k)",
        "                 continue",
        "             result[canonical] = v",
        "@@ -95,8 +103,17 @@ class TransactionNormaliser:",
        "             if isinstance(val, str):",
        '                 result["created_at"] = datetime.fromisoformat(val.replace("Z", "+00:00"))',
        '         if "amount_cents" in result:',
        '             result["amount_cents"] = int(result["amount_cents"])',
        '+        if "settlement_date" in result:',
        '+            val = result["settlement_date"]',
        "+            if isinstance(val, (int, float)):",
        '+                result["settlement_date"] = datetime.fromtimestamp(val, tz=timezone.utc).date()',
        "+            elif isinstance(val, str):",
        '+                result["settlement_date"] = datetime.fromisoformat(val).date()',
        '+        if "value_date" in result:',
        '+            val = result["value_date"]',
        "+            if isinstance(val, str):",
        '+                result["value_date"] = datetime.fromisoformat(val).date()',
        "         return result",
        " ",
        "diff --git a/src/pipeline/schema.py b/src/pipeline/schema.py",
        "index c1d8e22..f9a03b1 100644",
        "--- a/src/pipeline/schema.py",
        "+++ b/src/pipeline/schema.py",
        "@@ -1,5 +1,5 @@",
        " #!/usr/bin/env python3",
        '-"""Data schema definitions -- version 2.1.0"""',
        '+"""Data schema definitions -- version 2.2.0"""',
        " ",
        " from datetime import date, datetime",
        " from typing import Optional",
        "@@ -14,6 +14,9 @@ class NormalisedTransaction(BaseModel):",
        "     amount_cents: int",
        "     currency_code: str",
        "     transaction_type: Optional[str] = None",
        "+    settlement_date: Optional[date] = None",
        "+    value_date: Optional[date] = None",
        "+    merchant_name: Optional[str] = None",
        "+    merchant_category: Optional[str] = None",
        "     created_at: datetime",
        "     settled_at: Optional[datetime] = None",
        "     reference_note: Optional[str] = None",
        "@@ -28,6 +31,7 @@ class NormalisedTransaction(BaseModel):",
        " class EnrichedTransaction(NormalisedTransaction):",
        "     counterparty_name: Optional[str] = None",
        "     counterparty_country: Optional[str] = None",
        "+    counterparty_risk_tier: Optional[str] = None",
        "     fx_rate_to_usd: Optional[float] = None",
        "     risk_score: Optional[float] = None",
        "     risk_flags: list = []",
        "diff --git a/tests/test_normaliser.py b/tests/test_normaliser.py",
        "index 3b7f910..d2e1fe4 100644",
        "--- a/tests/test_normaliser.py",
        "+++ b/tests/test_normaliser.py",
        "@@ -1,5 +1,6 @@",
        " import pytest",
        " from datetime import date, datetime, timezone",
        "+from unittest.mock import patch, MagicMock",
        " from src.pipeline.normaliser import TransactionNormaliser",
        " from src.pipeline.schema import NormalisedTransaction",
        " ",
        "@@ -45,3 +46,18 @@ class TestTransactionNormaliser:",
        '         result = normaliser.normalise({"transaction_id": "t1", "accountId": "a1",',
        '                                        "amountCents": "1500", "currencyCode": "USD",',
        '                                        "createdAt": "2025-01-01T00:00:00Z"})',
        "+",
        "     def test_settlement_date_coercion(self, normaliser):",
        '+        raw = {**BASE_RECORD, "settlementDate": "2025-03-20"}',
        "         result = normaliser.normalise(raw)",
        "         assert result is not None",
        "         assert result.settlement_date == date(2025, 3, 20)",
        "+",
        "     def test_merchant_fields_mapped(self, normaliser):",
        '+        raw = {**BASE_RECORD, "merchantName": "ACME Corp", "merchantCategory": "retail"}',
        "         result = normaliser.normalise(raw)",
        "         assert result is not None",
        '         assert result.merchant_name == "ACME Corp"',
        '         assert result.merchant_category == "retail"',
        "+",
        "     def test_normalised_count_increments(self, normaliser):",
        "         normaliser.normalise(BASE_RECORD)",
        "         assert normaliser._normalised_count == 1",
    ]
    return "\n".join(lines)


TC5_GIT_DIFF = _build_git_diff()

# ---------------------------------------------------------------------------
# Test Case 6: Grep/ripgrep search results (~50 results)
# Expected: SearchCrunch groups by file, merges consecutive line numbers, RLE
# ---------------------------------------------------------------------------
def _build_search_results() -> str:
    """Build realistic grep -rn output for 50+ matches across several files."""
    lines = []
    # Matches in normaliser.py
    for ln in [45, 46, 47, 48, 49, 50, 51, 52, 87, 88, 134, 135, 136, 201, 202]:
        lines.append(f"src/pipeline/normaliser.py:{ln}:    transaction_id = record.get('transaction_id')")
    # Matches in schema.py
    for ln in [14, 15, 16, 28, 29, 30, 31, 32, 33]:
        lines.append(f"src/pipeline/schema.py:{ln}:    transaction_id: str")
    # Matches in tests
    for ln in [12, 13, 14, 34, 35, 46, 47, 48, 50, 51]:
        lines.append(f"tests/test_normaliser.py:{ln}:    transaction_id = 'txn_{ln:04d}'")
    for ln in [8, 9, 22, 23, 44, 45]:
        lines.append(f"tests/test_pipeline.py:{ln}:    transaction_id = 'test_txn_{ln}'")
    # Matches in enrichment
    for ln in [67, 68, 69, 70, 88, 89, 90, 102]:
        lines.append(f"src/pipeline/enrichment.py:{ln}:    key = f'enrich:{{record.transaction_id}}'")
    # Matches in metrics
    for ln in [33, 34, 55]:
        lines.append(f"src/pipeline/metrics.py:{ln}:    labels = dict(transaction_id=txn_id)")
    # Matches in config
    for ln in [19, 20]:
        lines.append(f"src/pipeline/config.py:{ln}:    transaction_id_prefix: str = 'txn'")
    # A separator line (should be filtered)
    lines.append("--")
    # More matches in a deeply nested util file
    for ln in [7, 8, 9, 201, 202, 203, 204]:
        lines.append(f"src/pipeline/utils/id_generator.py:{ln}:    return f'txn_{{uuid4().hex[:16]}}'")
    return "\n".join(lines)


TC6_SEARCH_RESULTS = _build_search_results()


# ---------------------------------------------------------------------------
# Master test cases list
# ---------------------------------------------------------------------------
# Each tuple: (display_name, content_type, language_hint_or_None, text)
TEST_CASES: list[tuple[str, str, str | None, str]] = [
    ("Python source (~200 lines)",   "code",   "python", TC1_PYTHON_SOURCE),
    ("JSON API response (100 items)", "json",   None,     TC2_JSON_API_RESPONSE),
    ("Build log (~500 lines)",        "log",    None,     TC3_BUILD_LOG),
    ("Agent conversation (8 msgs)",   "text",   None,     TC4_CONVERSATION),
    ("Git diff (~100 lines)",         "diff",   None,     TC5_GIT_DIFF),
    ("Search results (~50 matches)",  "search", None,     TC6_SEARCH_RESULTS),
]


# ---------------------------------------------------------------------------
# OLD compression path
# ---------------------------------------------------------------------------

def compress_old(text: str) -> str:
    """Run the old compressed_context.py ultra-compression path."""
    result = compress_with_stats(text, level="ultra")
    return result["compressed"]


# ---------------------------------------------------------------------------
# NEW FusionEngine path (uses individual stages if engine.py absent)
# ---------------------------------------------------------------------------

def _build_fallback_pipeline(content_type: str, language: str | None) -> FusionPipeline:
    """Build a per-content-type FusionPipeline from individual stages.

    Used when lib/fusion/engine.py does not exist yet.
    """
    stages = []

    if content_type == "code":
        stages.append(Neurosyntax())
    elif content_type == "json":
        stages.append(Ionizer())
    elif content_type == "log":
        stages.append(LogCrunch())
    elif content_type == "diff":
        stages.append(DiffCrunch())
    elif content_type == "search":
        stages.append(SearchCrunch())
    elif content_type == "text":
        stages.append(NexusStage())

    return FusionPipeline(stages)


def compress_new(
    text: str,
    content_type: str,
    language: str | None,
) -> tuple[str, str]:
    """Run the new FusionEngine (or per-stage fallback) and return (compressed, method)."""
    if _FUSION_ENGINE_AVAILABLE and FusionEngine is not None:
        engine = FusionEngine()
        result = engine.compress(text, content_type=content_type)
        compressed = result.get("compressed", result.get("content", text))
        return compressed, "FusionEngine"

    # FusionEngine not yet available — use individual stages directly.
    ctx = FusionContext(
        content=text,
        content_type=content_type,
        language=language,
    )
    pipeline = _build_fallback_pipeline(content_type, language)
    pipeline_result = pipeline.run(ctx)
    return pipeline_result.content, "FusionPipeline (stages)"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks() -> list[dict[str, Any]]:
    """Run all test cases through both compression paths and collect metrics."""
    results: list[dict[str, Any]] = []

    for name, content_type, language, text in TEST_CASES:
        orig_tokens = estimate_tokens(text)
        orig_chars = len(text)

        # --- OLD path ---
        t0 = time.perf_counter()
        old_compressed = compress_old(text)
        old_ms = (time.perf_counter() - t0) * 1000
        old_tokens = estimate_tokens(old_compressed)
        old_reduction = 100.0 * (1 - old_tokens / orig_tokens) if orig_tokens else 0.0

        # --- NEW path ---
        new_tokens: int | str
        new_reduction: float | str
        new_ms: float | str
        new_method: str

        try:
            t0 = time.perf_counter()
            new_compressed, new_method = compress_new(text, content_type, language)
            new_ms = (time.perf_counter() - t0) * 1000
            new_tokens = estimate_tokens(new_compressed)
            new_reduction = 100.0 * (1 - new_tokens / orig_tokens) if orig_tokens else 0.0
        except Exception as exc:
            new_tokens = "ERR"
            new_reduction = "ERR"
            new_ms = "ERR"
            new_method = f"error: {exc}"

        results.append({
            "name": name,
            "content_type": content_type,
            "original_tokens": orig_tokens,
            "original_chars": orig_chars,
            "old_tokens": old_tokens,
            "old_reduction_pct": old_reduction,
            "old_ms": old_ms,
            "new_tokens": new_tokens,
            "new_reduction_pct": new_reduction,
            "new_ms": new_ms,
            "new_method": new_method,
        })

    return results


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _fmt(value: Any, fmt: str = "") -> str:
    """Format a value, returning 'N/A' or 'ERR' strings as-is."""
    if isinstance(value, str):
        return value
    if fmt:
        return format(value, fmt)
    return str(value)


def print_table(results: list[dict[str, Any]]) -> None:
    """Print a rich comparison table to stdout."""
    divider = "=" * 110
    thin = "-" * 110

    print()
    print(divider)
    print("  FUSION BENCHMARK — OLD (compressed_context ultra) vs NEW (FusionPipeline stages)")
    if _FUSION_ENGINE_AVAILABLE:
        print("  NEW path: lib/fusion/engine.FusionEngine")
    else:
        print("  NEW path: individual FusionPipeline stages (engine.py not yet present)")
    print(divider)

    # Header
    col_name    = 36
    col_orig    = 10
    col_old_t   = 10
    col_old_pct = 9
    col_old_ms  = 9
    col_new_t   = 10
    col_new_pct = 9
    col_new_ms  = 9
    col_delta   = 10

    header = (
        f"{'Test Case':<{col_name}}"
        f"{'Orig Tok':>{col_orig}}"
        f"{'Old Tok':>{col_old_t}}"
        f"{'Old %Red':>{col_old_pct}}"
        f"{'Old ms':>{col_old_ms}}"
        f"{'New Tok':>{col_new_t}}"
        f"{'New %Red':>{col_new_pct}}"
        f"{'New ms':>{col_new_ms}}"
        f"{'Delta Tok':>{col_delta}}"
    )
    print(header)
    print(thin)

    for r in results:
        name_trunc = r["name"][:col_name - 1]
        orig = r["original_tokens"]

        old_t   = _fmt(r["old_tokens"])
        old_pct = _fmt(r["old_reduction_pct"], ".1f") + "%" if isinstance(r["old_reduction_pct"], float) else r["old_reduction_pct"]
        old_ms  = _fmt(r["old_ms"], ".0f") if isinstance(r["old_ms"], float) else r["old_ms"]

        new_t   = _fmt(r["new_tokens"])
        new_pct = _fmt(r["new_reduction_pct"], ".1f") + "%" if isinstance(r["new_reduction_pct"], float) else r["new_reduction_pct"]
        new_ms  = _fmt(r["new_ms"], ".0f") if isinstance(r["new_ms"], float) else r["new_ms"]

        # Delta: positive = new saves more tokens than old
        if isinstance(r["new_tokens"], int) and isinstance(r["old_tokens"], int):
            delta = r["old_tokens"] - r["new_tokens"]
            delta_str = f"+{delta}" if delta > 0 else str(delta)
        else:
            delta_str = "N/A"

        print(
            f"{name_trunc:<{col_name}}"
            f"{orig:>{col_orig},}"
            f"{old_t:>{col_old_t}}"
            f"{old_pct:>{col_old_pct}}"
            f"{old_ms:>{col_old_ms}}"
            f"{new_t:>{col_new_t}}"
            f"{new_pct:>{col_new_pct}}"
            f"{new_ms:>{col_new_ms}}"
            f"{delta_str:>{col_delta}}"
        )

    print(thin)

    # Totals row
    orig_total = sum(r["original_tokens"] for r in results)
    old_total  = sum(r["old_tokens"] for r in results if isinstance(r["old_tokens"], int))
    new_total  = sum(r["new_tokens"] for r in results if isinstance(r["new_tokens"], int))

    old_total_pct = 100.0 * (1 - old_total / orig_total) if orig_total else 0.0
    new_total_pct = 100.0 * (1 - new_total / orig_total) if orig_total else 0.0
    total_delta = old_total - new_total

    print(
        f"{'TOTAL':<{col_name}}"
        f"{orig_total:>{col_orig},}"
        f"{old_total:>{col_old_t},}"
        f"{old_total_pct:>{col_old_pct - 1}.1f}%"
        f"{'':>{col_old_ms}}"
        f"{new_total:>{col_new_t},}"
        f"{new_total_pct:>{col_new_pct - 1}.1f}%"
        f"{'':>{col_new_ms}}"
        f"{('+' + str(total_delta) if total_delta > 0 else str(total_delta)):>{col_delta}}"
    )
    print(divider)

    # Per-case notes
    print()
    print("  Notes:")
    notes = {
        "Python source (~200 lines)":    "Neurosyntax: removes comments/docstrings, collapses blank lines",
        "JSON API response (100 items)": "Ionizer: samples ~20 items, keeps first/last + error records",
        "Build log (~500 lines)":        "LogCrunch: preserves ERROR/WARN/stack traces, collapses repeated INFO",
        "Agent conversation (8 msgs)":   "NexusStage: stopword removal + repeated-ngram dedup (fallback path)",
        "Git diff (~100 lines)":         "DiffCrunch: keeps +/- lines, compresses context blocks to 1 line each end",
        "Search results (~50 matches)":  "SearchCrunch: groups by file, merges consecutive line ranges",
    }
    for r in results:
        note = notes.get(r["name"], "")
        if note:
            print(f"  [{r['name']}]")
            print(f"    {note}")
    print()

    # Stage pipeline detail
    print("  Stage detail (NEW path — stages active per content_type):")
    stage_map = {
        "code":   "Neurosyntax (order=25)",
        "json":   "Ionizer (order=15)",
        "log":    "LogCrunch (order=16)",
        "text":   "NexusStage (order=35) — torch fallback rule-based",
        "diff":   "DiffCrunch (order=18)",
        "search": "SearchCrunch (order=17)",
    }
    for r in results:
        stage = stage_map.get(r["content_type"], "—")
        print(f"  {r['content_type']:<8} -> {stage}")
    print()

    print("  Delta Tok: positive = NEW saves more tokens than OLD vs original")
    print(divider)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Running FusionEngine benchmark harness...")
    print(f"  Test cases: {len(TEST_CASES)}")
    print(f"  OLD path:   compressed_context.compress_with_stats(level='ultra')")
    print(f"  NEW path:   {'lib/fusion/engine.FusionEngine' if _FUSION_ENGINE_AVAILABLE else 'FusionPipeline (per-stage, engine.py absent)'}")
    print()

    results = run_benchmarks()
    print_table(results)


if __name__ == "__main__":
    main()

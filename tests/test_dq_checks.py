"""Unit tests for `dq/checks.py` and `dq/gate.py` on small synthetic fixtures.

The fixtures mimic the upstream anomalies listed in `docs/STUDENT_ASSIGNMENT.md`
(Section 3.5), so the DQ suite can be developed without the R2 landing dataset.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pyspark.sql import types as T

from dq import checks
from dq.framework import DQ_AUDIT_SCHEMA, DQCheckResult, DQGateFailedError, enforce_dq_gate
from dq.gate import collect_layer_results, missing_audit_results

CTX = {"run_id": "test-run", "batch_date": "2026-01-05"}  # Monday


def _d(value: str) -> date:
    return date.fromisoformat(value)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def timesheets_df(spark):
    rows = [
        ("TS1", "E1", "PRJ_0001", _d("2026-01-05"), 8.0),
        ("TS1", "E1", "PRJ_0001", _d("2026-01-05"), 8.0),  # exact duplicate
        ("TS2", "E1", "PRJ_0001", _d("2026-01-05"), 7.5),
        ("TS2", "E1", "PRJ_0001", _d("2026-01-05"), 7.0),  # near duplicate
        ("TS3", "E2", None, _d("2026-01-05"), 6.0),  # NULL project_id
        ("TS4", "E2", "PRJ_0002", _d("2026-01-05"), 0.0),  # hours <= 0
        ("TS5", "E3", "PRJ_0002", _d("2026-01-05"), 25.0),  # hours > 24
        ("TS6", "E4", "PRJ_0003", _d("2026-01-05"), 8.0),  # zombie (terminated)
    ]
    schema = "ts_id string, emp_id string, project_id string, work_date date, hours double"
    return spark.createDataFrame(rows, schema)


@pytest.fixture()
def scd2_df(spark):
    rows = [
        # E1: clean history (closed intervals)
        ("E1", _d("2026-01-01"), _d("2026-02-14"), False),
        ("E1", _d("2026-02-15"), None, True),
        # E2: overlapping versions
        ("E2", _d("2026-01-01"), _d("2026-03-01"), False),
        ("E2", _d("2026-02-15"), None, True),
        # E3: two current versions with the same valid_from (duplicate key)
        ("E3", _d("2026-01-01"), None, True),
        ("E3", _d("2026-01-01"), None, True),
    ]
    schema = "emp_id string, valid_from date, valid_to date, is_current boolean"
    return spark.createDataFrame(rows, schema)


# ---------------------------------------------------------------------------
# Hard checks
# ---------------------------------------------------------------------------
def test_expected_schema(spark, timesheets_df) -> None:
    expected = {
        "ts_id": T.StringType(),
        "hours": T.DoubleType(),
        "work_date": T.DateType(),
    }
    ok = checks.check_expected_schema(
        timesheets_df, expected, layer="silver", table_name="silver_timesheets", **CTX
    )
    assert (ok.status, ok.failed_count, ok.total_count) == ("PASS", 0, 3)

    wrong = {**expected, "hours": T.IntegerType(), "missing_col": T.StringType()}
    bad = checks.check_expected_schema(
        timesheets_df, wrong, layer="silver", table_name="silver_timesheets", **CTX
    )
    assert (bad.status, bad.failed_count, bad.total_count) == ("FAIL", 2, 4)


def test_unique_key_counts_surplus_rows(timesheets_df) -> None:
    res = checks.check_unique_key(
        timesheets_df, ["ts_id"], layer="silver", table_name="silver_timesheets", **CTX
    )
    assert res.check_name == "unique_ts_id"
    assert (res.status, res.failed_count, res.total_count) == ("FAIL", 2, 8)

    deduped = timesheets_df.dropDuplicates(["ts_id"])
    ok = checks.check_unique_key(
        deduped, ["ts_id"], layer="silver", table_name="silver_timesheets", **CTX
    )
    assert (ok.status, ok.failed_count) == ("PASS", 0)


def test_non_empty_partition_handles_weekend_fx(spark) -> None:
    fx = spark.createDataFrame(
        [(_d("2026-01-02"), "EUR", "USD", 1.1)],
        "_batch_date date, currency_from string, currency_to string, rate double",
    )
    common = {"run_id": "r", "layer": "bronze", "table_name": "fx_rates"}

    friday = checks.check_non_empty_partition(fx, batch_date="2026-01-02", **common)
    assert (friday.status, friday.total_count) == ("PASS", 1)

    saturday_strict = checks.check_non_empty_partition(fx, batch_date="2026-01-03", **common)
    assert saturday_strict.status == "FAIL"

    saturday_fx = checks.check_non_empty_partition(
        fx, batch_date="2026-01-03", allow_weekends_empty=True, **common
    )
    assert (saturday_fx.status, saturday_fx.failed_count) == ("PASS", 0)

    monday_missing = checks.check_non_empty_partition(
        fx, batch_date="2026-01-05", allow_weekends_empty=True, **common
    )
    assert monday_missing.status == "FAIL"


def test_scd2_no_overlaps_and_single_current(scd2_df) -> None:
    overlaps = checks.check_scd2_no_overlaps(scd2_df, **CTX)
    # E2 first version overlaps its successor; E3 duplicate open-ended versions overlap.
    assert (overlaps.status, overlaps.failed_count, overlaps.total_count) == ("FAIL", 2, 6)

    clean = scd2_df.filter("emp_id = 'E1'")
    assert checks.check_scd2_no_overlaps(clean, **CTX).status == "PASS"

    # Half-open convention: valid_to == next valid_from is allowed.
    half_open_df = clean.sparkSession.createDataFrame(
        [("E9", _d("2026-01-01"), _d("2026-02-15")), ("E9", _d("2026-02-15"), None)],
        "emp_id string, valid_from date, valid_to date",
    )
    assert checks.check_scd2_no_overlaps(half_open_df, half_open=True, **CTX).status == "PASS"
    assert checks.check_scd2_no_overlaps(half_open_df, **CTX).status == "FAIL"

    current = checks.check_scd2_single_current(scd2_df, **CTX)
    assert (current.status, current.failed_count, current.total_count) == ("FAIL", 1, 3)

    unique = checks.check_unique_key(
        scd2_df,
        ["emp_id", "valid_from"],
        layer="silver",
        table_name="silver_employees_scd2",
        **CTX,
    )
    assert (unique.check_name, unique.failed_count) == ("unique_emp_id_valid_from", 1)


def test_zero_null_eur(spark) -> None:
    df = spark.createDataFrame(
        [("I1", 100.0), ("I2", -20.0), ("I3", None)], "inv_id string, amount_eur double"
    )
    res = checks.check_zero_null_eur(
        df, "amount_eur", layer="silver", table_name="silver_invoices", **CTX
    )
    assert (res.check_name, res.status, res.failed_count) == ("zero_null_amount_eur", "FAIL", 1)


def test_reconciliation_detects_silent_loss() -> None:
    ok = checks.check_reconciliation(
        8,
        valid_count=2,
        quarantined_count=4,
        deduplicated_count=2,
        table_name="silver_timesheets",
        **CTX,
    )
    assert ok.status == "PASS"
    lost = checks.check_reconciliation(
        8,
        valid_count=2,
        quarantined_count=3,
        deduplicated_count=2,
        table_name="silver_timesheets",
        **CTX,
    )
    assert (lost.status, lost.failed_count) == ("FAIL", 1)


# ---------------------------------------------------------------------------
# Soft checks
# ---------------------------------------------------------------------------
def test_quarantine_ratio_threshold() -> None:
    ok = checks.check_quarantine_ratio(92, 8, table_name="silver_timesheets", **CTX)
    assert (ok.status, ok.failed_count, ok.total_count) == ("PASS", 8, 100)
    assert ok.check_name == "quarantine_ratio_le_8pct"

    warn = checks.check_quarantine_ratio(91, 9, table_name="silver_timesheets", **CTX)
    assert (warn.status, warn.severity) == ("WARN", "WARN")

    empty = checks.check_quarantine_ratio(0, 0, table_name="silver_timesheets", **CTX)
    assert empty.status == "PASS"


def test_duplicate_rate(timesheets_df) -> None:
    res = checks.check_duplicate_rate(
        timesheets_df, ["ts_id"], max_ratio=0.10, table_name="timesheets", **CTX
    )
    assert (res.status, res.failed_count, res.total_count) == ("WARN", 2, 8)


def test_referential_integrity_orphan_invoices(spark) -> None:
    invoices = spark.createDataFrame(
        [("I1", "PRJ_0001"), ("I2", "PRJ_ORPHAN_1"), ("I3", None), ("I4", "PRJ_0002")],
        "inv_id string, project_id string",
    )
    projects = spark.createDataFrame([("PRJ_0001",), ("PRJ_0002",)], "project_id string")
    res = checks.check_referential_integrity(
        invoices, projects, "project_id", table_name="silver_invoices", **CTX
    )
    assert (res.check_name, res.status, res.failed_count) == ("fk_project_id_exists", "WARN", 1)

    tolerant = checks.check_referential_integrity(
        invoices, projects, "project_id", max_ratio=0.5, table_name="silver_invoices", **CTX
    )
    assert tolerant.status == "PASS"


def test_row_condition_zombie_timesheets(spark, timesheets_df) -> None:
    employees = spark.createDataFrame(
        [("E1", None), ("E4", _d("2026-01-03"))], "emp_id string, termination_date date"
    )
    joined = timesheets_df.join(employees, "emp_id", "left")
    res = checks.check_row_condition(
        joined,
        "work_date >= termination_date",
        check_name="zombie_timesheets",
        layer="silver",
        table_name="silver_timesheets",
        **CTX,
    )
    # NULL termination_date must not count as a violation.
    assert (res.status, res.failed_count, res.total_count) == ("WARN", 1, 8)


def test_accepted_values_and_value_range(spark, timesheets_df) -> None:
    hours = checks.check_value_range(
        timesheets_df,
        "hours",
        min_value=0.01,
        max_value=24,
        layer="silver",
        table_name="silver_timesheets",
        **CTX,
    )
    assert (hours.check_name, hours.status, hours.failed_count) == ("range_hours", "FAIL", 2)

    answers = spark.createDataFrame(
        [("Q01", 11), ("Q01", 9), ("Q02", -1), ("Q02", 3), ("Q03", None)],
        "question_id string, numeric_score int",
    )
    scores = checks.check_value_range(
        answers.filter("question_id <> 'Q01'"),
        "numeric_score",
        min_value=1,
        max_value=5,
        layer="silver",
        table_name="silver_survey_answers",
        severity="WARN",
        **CTX,
    )
    assert (scores.status, scores.failed_count) == ("WARN", 1)

    tiers = spark.createDataFrame([("Gold",), ("Platinum",), (None,)], "tier string")
    loose = checks.check_accepted_values(
        tiers,
        "tier",
        ["Standard", "Silver", "Gold", "Enterprise"],
        layer="silver",
        table_name="silver_clients",
        **CTX,
    )
    strict = checks.check_accepted_values(
        tiers,
        "tier",
        ["Standard", "Silver", "Gold", "Enterprise"],
        allow_null=False,
        layer="silver",
        table_name="silver_clients",
        **CTX,
    )
    assert (loose.failed_count, strict.failed_count) == (1, 2)


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------
def test_layer_gate_fails_on_missing_audit_and_hard_failures(spark) -> None:
    now = datetime.now(timezone.utc)
    audit_rows = [
        (
            "r1",
            _d("2026-01-05"),
            "silver",
            "silver_clients",
            "not_null_client_id",
            "ERROR",
            "PASS",
            0,
            300,
            now,
        ),
        (
            "r1",
            _d("2026-01-05"),
            "silver",
            "silver_timesheets",
            "quarantine_ratio_le_8pct",
            "WARN",
            "WARN",
            9,
            100,
            now,
        ),
        (
            "r1",
            _d("2026-01-04"),
            "silver",
            "silver_invoices",
            "zero_null_amount_eur",
            "ERROR",
            "FAIL",
            1,
            10,
            now,
        ),  # other batch_date -> must be ignored
    ]
    audit_df = spark.createDataFrame(audit_rows, DQ_AUDIT_SCHEMA)

    existing = collect_layer_results(audit_df, "2026-01-05", "silver")
    assert {r.table_name for r in existing} == {"silver_clients", "silver_timesheets"}
    assert all(isinstance(r, DQCheckResult) for r in existing)

    # WARN alone passes the gate.
    enforce_dq_gate(existing)

    missing = missing_audit_results(
        existing,
        ["silver_clients", "silver_timesheets", "silver_invoices"],
        run_id="gate",
        batch_date="2026-01-05",
        layer="silver",
    )
    assert [(m.table_name, m.check_name, m.status) for m in missing] == [
        ("silver_invoices", "audit_present", "FAIL")
    ]
    with pytest.raises(DQGateFailedError, match="silver_invoices.audit_present"):
        enforce_dq_gate([*existing, *missing])

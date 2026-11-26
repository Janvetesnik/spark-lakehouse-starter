"""Reusable Bronze / Silver / Gold Data Quality checks.

Every check returns a `DQCheckResult` (see `dq.framework`) that is persisted to
`dq_audit_results` via `write_audit_results` and evaluated by `enforce_dq_gate`.

Status semantics (shared by all checks):
- `failed_count == 0` (or ratio within threshold)  -> `PASS`
- violation and `severity == "WARN"`               -> `WARN`  (soft check, pipeline continues)
- violation and `severity == "ERROR"`              -> `FAIL`  (hard check, gate raises)

Hard checks (ERROR) — Section 4.3 of `docs/STUDENT_ASSIGNMENT.md`:
    check_expected_schema, check_not_null (framework), check_unique_key,
    check_non_empty_partition, check_scd2_no_overlaps, check_scd2_single_current,
    check_zero_null_eur, check_reconciliation
Soft checks (WARN):
    check_quarantine_ratio, check_duplicate_rate, check_referential_integrity,
    check_ratio_threshold (orphan rate, missing comments, ...), check_row_condition
    (zombie timesheets, ...), check_accepted_values, check_value_range
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, StructType

from dq.framework import DQCheckResult, _parse_batch_date

OPEN_END_DATE = date(9999, 12, 31)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _status(violated: bool, severity: str) -> str:
    if not violated:
        return "PASS"
    return "WARN" if severity.upper() == "WARN" else "FAIL"


def _result(
    *,
    check_name: str,
    failed_count: int,
    total_count: int,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    severity: str,
    violated: bool | None = None,
) -> DQCheckResult:
    """Build a `DQCheckResult`; `violated` defaults to `failed_count > 0`."""
    is_violated = failed_count > 0 if violated is None else violated
    return DQCheckResult(
        run_id=run_id,
        batch_date=_parse_batch_date(batch_date),
        layer=layer,
        table_name=table_name,
        check_name=check_name,
        severity=severity.upper(),
        status=_status(is_violated, severity),
        failed_count=int(failed_count),
        total_count=int(total_count),
        checked_at=datetime.now(timezone.utc),
    )


def _ratio_suffix(max_ratio: float) -> str:
    return f"{max_ratio * 100:g}pct".replace(".", "_")


# ---------------------------------------------------------------------------
# Hard checks (default severity ERROR)
# ---------------------------------------------------------------------------
def check_expected_schema(
    df: DataFrame,
    expected: StructType | Mapping[str, DataType],
    *,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    severity: str = "ERROR",
) -> DQCheckResult:
    """Verify that every expected column exists with the expected Spark data type.

    `failed_count` = number of expected columns that are missing or mistyped,
    `total_count` = number of expected columns. Extra columns are allowed.
    """
    expected_types = (
        {f.name: f.dataType for f in expected.fields}
        if isinstance(expected, StructType)
        else dict(expected)
    )
    actual_types = {f.name: f.dataType for f in df.schema.fields}
    failed = sum(1 for name, dtype in expected_types.items() if actual_types.get(name) != dtype)
    return _result(
        check_name="expected_schema",
        failed_count=failed,
        total_count=len(expected_types),
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_unique_key(
    df: DataFrame,
    key_columns: Sequence[str],
    *,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    severity: str = "ERROR",
) -> DQCheckResult:
    """Verify uniqueness of a primary/composite key (e.g. `(emp_id, valid_from)`).

    `failed_count` = surplus rows, i.e. `total_rows - distinct_keys`.
    """
    keys = list(key_columns)
    if not keys:
        raise ValueError("key_columns must contain at least one column.")
    total = df.count()
    distinct_keys = df.select(*keys).distinct().count()
    return _result(
        check_name=f"unique_{'_'.join(keys)}",
        failed_count=total - distinct_keys,
        total_count=total,
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_non_empty_partition(
    df: DataFrame,
    *,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    allow_weekends_empty: bool = False,
    severity: str = "ERROR",
    partition_column: str = "_batch_date",
) -> DQCheckResult:
    """Verify the `batch_date` partition contains > 0 rows.

    If `partition_column` exists in `df`, only rows of `batch_date` are counted;
    otherwise `df` is treated as the already-filtered partition. With
    `allow_weekends_empty=True` (weekday-only `fx_rates`) an empty Saturday/Sunday
    partition passes.
    """
    parsed_date = _parse_batch_date(batch_date)
    scoped = (
        df.filter(F.col(partition_column) == F.lit(parsed_date))
        if partition_column in df.columns
        else df
    )
    row_count = scoped.count()
    is_weekend = parsed_date.weekday() >= 5
    violated = row_count == 0 and not (allow_weekends_empty and is_weekend)
    return _result(
        check_name="non_empty_partition",
        failed_count=1 if violated else 0,
        total_count=row_count,
        run_id=run_id,
        batch_date=parsed_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_scd2_no_overlaps(
    df: DataFrame,
    *,
    run_id: str,
    batch_date: str | date,
    layer: str = "silver",
    table_name: str = "silver_employees_scd2",
    severity: str = "ERROR",
    key_column: str = "emp_id",
    valid_from_column: str = "valid_from",
    valid_to_column: str = "valid_to",
    half_open: bool = False,
) -> DQCheckResult:
    """Verify no overlapping validity intervals per key in an SCD Type 2 table.

    Intervals are closed `[valid_from, valid_to]` by default (next version starts
    the day after `valid_to`); set `half_open=True` for `[valid_from, valid_to)`.
    A NULL `valid_to` means open-ended (current version). Rows with
    `valid_from > valid_to` are counted as violations as well.

    `failed_count` = number of versions that overlap their successor or are inverted.
    """
    end_col = F.coalesce(F.col(valid_to_column), F.lit(OPEN_END_DATE))
    window = Window.partitionBy(key_column).orderBy(F.col("_from").asc(), F.col("_to").asc())
    with_next = df.select(
        F.col(key_column),
        F.col(valid_from_column).alias("_from"),
        end_col.alias("_to"),
    ).withColumn("_next_from", F.lead("_from").over(window))

    overlap = (
        F.col("_next_from") < F.col("_to") if half_open else F.col("_next_from") <= F.col("_to")
    )
    inverted = F.col("_from") > F.col("_to")
    failed = with_next.filter((F.col("_next_from").isNotNull() & overlap) | inverted).count()
    return _result(
        check_name="scd2_no_overlaps",
        failed_count=failed,
        total_count=with_next.count(),
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_scd2_single_current(
    df: DataFrame,
    *,
    run_id: str,
    batch_date: str | date,
    layer: str = "silver",
    table_name: str = "silver_employees_scd2",
    severity: str = "ERROR",
    key_column: str = "emp_id",
    current_flag_column: str = "is_current",
) -> DQCheckResult:
    """Verify every key has exactly one `is_current = true` version.

    `failed_count` = keys with 0 or >1 current versions, `total_count` = distinct keys.
    """
    per_key = df.groupBy(key_column).agg(
        F.sum(F.when(F.col(current_flag_column), 1).otherwise(0)).alias("_n_current")
    )
    failed = per_key.filter(F.col("_n_current") != 1).count()
    return _result(
        check_name="scd2_single_current",
        failed_count=failed,
        total_count=per_key.count(),
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_zero_null_eur(
    df: DataFrame,
    eur_column: str,
    *,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    severity: str = "ERROR",
) -> DQCheckResult:
    """Verify zero NULL EUR amounts after FX conversion (invoices & expenses)."""
    total = df.count()
    failed = df.filter(F.col(eur_column).isNull()).count()
    return _result(
        check_name=f"zero_null_{eur_column}",
        failed_count=failed,
        total_count=total,
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_reconciliation(
    source_count: int,
    valid_count: int,
    quarantined_count: int,
    deduplicated_count: int = 0,
    *,
    run_id: str,
    batch_date: str | date,
    layer: str = "silver",
    table_name: str,
    severity: str = "ERROR",
) -> DQCheckResult:
    """Verify zero silent data loss: `source = valid + quarantined + deduplicated`.

    `failed_count` = absolute difference (rows lost or invented), `total_count` = source rows.
    """
    accounted = valid_count + quarantined_count + deduplicated_count
    return _result(
        check_name="reconciliation_no_silent_loss",
        failed_count=abs(source_count - accounted),
        total_count=source_count,
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Soft checks (default severity WARN)
# ---------------------------------------------------------------------------
def check_ratio_threshold(
    failed_count: int,
    total_count: int,
    *,
    check_name: str,
    max_ratio: float,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    severity: str = "WARN",
) -> DQCheckResult:
    """Generic rate metric: violated when `failed_count / total_count > max_ratio`.

    The raw counts are always logged so the reporting layer can chart the rate.
    """
    ratio = failed_count / total_count if total_count else 0.0
    return _result(
        check_name=check_name,
        failed_count=failed_count,
        total_count=total_count,
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
        violated=ratio > max_ratio,
    )


def check_quarantine_ratio(
    valid_count: int,
    quarantined_count: int,
    *,
    max_ratio: float = 0.08,
    run_id: str,
    batch_date: str | date,
    layer: str = "silver",
    table_name: str,
    severity: str = "WARN",
) -> DQCheckResult:
    """Emit WARN when quarantined_count / (valid_count + quarantined_count) > max_ratio."""
    return check_ratio_threshold(
        quarantined_count,
        valid_count + quarantined_count,
        check_name=f"quarantine_ratio_le_{_ratio_suffix(max_ratio)}",
        max_ratio=max_ratio,
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_duplicate_rate(
    df: DataFrame,
    key_columns: Sequence[str],
    *,
    max_ratio: float = 0.05,
    run_id: str,
    batch_date: str | date,
    layer: str = "bronze",
    table_name: str,
    severity: str = "WARN",
) -> DQCheckResult:
    """Log surplus duplicate rows by key (e.g. raw `ts_id`) and warn above `max_ratio`."""
    keys = list(key_columns)
    total = df.count()
    surplus = total - df.select(*keys).distinct().count()
    return check_ratio_threshold(
        surplus,
        total,
        check_name=f"duplicate_rate_{'_'.join(keys)}",
        max_ratio=max_ratio,
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_referential_integrity(
    df: DataFrame,
    ref_df: DataFrame,
    fk_column: str,
    ref_column: str | None = None,
    *,
    max_ratio: float = 0.0,
    run_id: str,
    batch_date: str | date,
    layer: str = "silver",
    table_name: str,
    severity: str = "WARN",
) -> DQCheckResult:
    """Count orphan foreign keys (`left_anti` join); NULL FKs are not counted as orphans.

    Example: invoices.project_id -> silver_projects.project_id (orphan invoice rate).
    """
    ref_key = ref_column or fk_column
    non_null = df.filter(F.col(fk_column).isNotNull())
    ref_keys = ref_df.select(F.col(ref_key).alias("_ref_key")).distinct()
    orphans = non_null.join(
        ref_keys, non_null[fk_column] == ref_keys["_ref_key"], "left_anti"
    ).count()
    return check_ratio_threshold(
        orphans,
        df.count(),
        check_name=f"fk_{fk_column}_exists",
        max_ratio=max_ratio,
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_row_condition(
    df: DataFrame,
    failing_condition: Column | str,
    *,
    check_name: str,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    severity: str = "WARN",
) -> DQCheckResult:
    """Count rows matching a *failing* condition (e.g. zombie timesheets:
    `work_date >= termination_date`, or missing comments: `has_written_comment = false`).
    """
    condition = (
        F.expr(failing_condition) if isinstance(failing_condition, str) else (failing_condition)
    )
    total = df.count()
    failed = df.filter(F.coalesce(condition, F.lit(False))).count()
    return _result(
        check_name=check_name,
        failed_count=failed,
        total_count=total,
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_accepted_values(
    df: DataFrame,
    column_name: str,
    accepted: Sequence[object],
    *,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    allow_null: bool = True,
    severity: str = "ERROR",
) -> DQCheckResult:
    """Verify `column_name` only contains values from `accepted` (domain check)."""
    col = F.col(column_name)
    failing = ~col.isin(*accepted)
    if not allow_null:
        failing = col.isNull() | failing
    return check_row_condition(
        df,
        failing,
        check_name=f"accepted_values_{column_name}",
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )


def check_value_range(
    df: DataFrame,
    column_name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    run_id: str,
    batch_date: str | date,
    layer: str,
    table_name: str,
    severity: str = "ERROR",
) -> DQCheckResult:
    """Verify `min_value <= column <= max_value` (inclusive; NULLs are ignored).

    Example: Gold `gross_margin_pct`, Silver survey `numeric_score` in `[scale_min, scale_max]`.
    """
    if min_value is None and max_value is None:
        raise ValueError("Provide at least one of min_value / max_value.")
    col = F.col(column_name)
    failing = F.lit(False)
    if min_value is not None:
        failing = failing | (col < F.lit(min_value))
    if max_value is not None:
        failing = failing | (col > F.lit(max_value))
    return check_row_condition(
        df,
        failing,
        check_name=f"range_{column_name}",
        run_id=run_id,
        batch_date=batch_date,
        layer=layer,
        table_name=table_name,
        severity=severity,
    )

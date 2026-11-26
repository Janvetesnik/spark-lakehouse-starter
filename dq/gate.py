"""Layer-level DQ gate for Airflow (`dq_bronze_gate`, `dq_silver_gate`, `dq_gold_gate`).

Each Spark job runs its own table checks and persists them to `dq_audit_results`.
This gate then reads the audit table for one `(batch_date, layer)` and fails when:
1. any `severity = ERROR` check has `status = FAIL`, or
2. a required table has no audit records for that batch (a check that did not run
   counts as a failure — "no silent skips").

Usage (inside the Spark / Airflow containers):
    spark-submit /opt/workspace/dq/gate.py --layer silver --batch-date 2026-01-01 \
        --run-id manual-test --tables silver_clients
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Sequence
from datetime import date, datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from dq.framework import (
    DQCheckResult,
    _parse_batch_date,
    enforce_dq_gate,
    write_audit_results,
)

# Tables that must have audit records for every daily batch, per layer.
REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "bronze": (
        "clients",
        "projects",
        "survey_questions",
        "employees",
        "timesheets",
        "fx_rates",
        "invoices",
        "expenses",
        "crm_events",
        "employee_surveys",
    ),
    "silver": (
        "silver_clients",
        "silver_employees_scd2",
        "silver_timesheets",
        "silver_fx_rates_daily",
        "silver_invoices",
        "silver_expenses",
        "silver_crm_events",
        "silver_survey_responses",
        "silver_survey_answers",
    ),
    "gold": (
        "gold_project_profitability",
        "gold_department_finance",
        "gold_client_engagement",
        "gold_hr_enps_analytics",
        "gold_employee_churn_features",
    ),
}


def collect_layer_results(
    audit_df: DataFrame, batch_date: str | date, layer: str
) -> list[DQCheckResult]:
    """Load audit rows for one `(batch_date, layer)` as `DQCheckResult` objects."""
    parsed_date = _parse_batch_date(batch_date)
    rows = audit_df.filter(
        (F.col("batch_date") == F.lit(parsed_date)) & (F.col("layer") == F.lit(layer))
    ).collect()
    return [
        DQCheckResult(
            run_id=r["run_id"],
            batch_date=r["batch_date"],
            layer=r["layer"],
            table_name=r["table_name"],
            check_name=r["check_name"],
            severity=r["severity"],
            status=r["status"],
            failed_count=int(r["failed_count"]),
            total_count=int(r["total_count"]),
            checked_at=r["checked_at"],
        )
        for r in rows
    ]


def missing_audit_results(
    existing: Sequence[DQCheckResult],
    required_tables: Sequence[str],
    *,
    run_id: str,
    batch_date: str | date,
    layer: str,
) -> list[DQCheckResult]:
    """Emit one ERROR/FAIL `audit_present` result per required table without audit rows."""
    audited = {r.table_name for r in existing}
    now = datetime.now(timezone.utc)
    parsed_date = _parse_batch_date(batch_date)
    return [
        DQCheckResult(
            run_id=run_id,
            batch_date=parsed_date,
            layer=layer,
            table_name=table,
            check_name="audit_present",
            severity="ERROR",
            status="FAIL",
            failed_count=1,
            total_count=1,
            checked_at=now,
        )
        for table in required_tables
        if table not in audited
    ]


def run_layer_gate(
    spark: SparkSession,
    *,
    audit_table_path: str,
    batch_date: str,
    layer: str,
    run_id: str,
    required_tables: Sequence[str] | None = None,
) -> list[DQCheckResult]:
    """Evaluate the gate; persists `audit_present` failures and raises on hard failures."""
    from delta.tables import DeltaTable

    tables = list(REQUIRED_TABLES[layer] if required_tables is None else required_tables)
    existing: list[DQCheckResult] = []
    if DeltaTable.isDeltaTable(spark, audit_table_path):
        existing = collect_layer_results(
            spark.read.format("delta").load(audit_table_path), batch_date, layer
        )

    missing = missing_audit_results(
        existing, tables, run_id=run_id, batch_date=batch_date, layer=layer
    )
    if missing:
        write_audit_results(spark, missing, audit_table_path)

    results = [*existing, *missing]
    print(
        f"DQ gate layer={layer} batch_date={batch_date}: "
        f"{sum(r.status == 'PASS' for r in results)} PASS, "
        f"{sum(r.status == 'WARN' for r in results)} WARN, "
        f"{sum(r.status == 'FAIL' for r in results)} FAIL"
    )
    enforce_dq_gate(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the DQ gate for one Medallion layer.")
    parser.add_argument("--layer", required=True, choices=sorted(REQUIRED_TABLES))
    parser.add_argument("--batch-date", required=True, help="Logical batch date (YYYY-MM-DD).")
    parser.add_argument("--run-id", default=None, help="Orchestrator run identifier.")
    parser.add_argument("--lakehouse-root", default=None, help="Override Lakehouse root path.")
    parser.add_argument(
        "--tables",
        default=None,
        help="Comma-separated required tables (default: all tables of the layer).",
    )
    return parser.parse_args()


def main() -> None:
    from spark_jobs.common.spark_session import get_lakehouse_root, get_spark_session, join_path

    args = parse_args()
    tables = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None
    spark = get_spark_session(f"dq-gate-{args.layer}")
    try:
        run_layer_gate(
            spark,
            audit_table_path=join_path(
                get_lakehouse_root(args.lakehouse_root), "dq", "dq_audit_results"
            ),
            batch_date=args.batch_date,
            layer=args.layer,
            run_id=args.run_id or f"manual-{args.batch_date}-{uuid.uuid4().hex[:8]}",
            required_tables=tables,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

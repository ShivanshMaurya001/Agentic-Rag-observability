
"""Single-page, read-only view of recorded RAG runs."""

import sqlite3

import altair as alt
import streamlit as st

import queries


MODES = (("baseline", "Baseline"), ("corrected", "Corrected"))
BASELINE_COLOR = "#8A8F98"
CORRECTED_COLOR = "#557A9E"


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _number(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _cost(value: float | str | None) -> str:
    if value in (None, 0, 0.0, queries.NOT_RECORDED):
        return queries.NOT_RECORDED
    return f"{value:.6f}"


def _truncate(value: str | None, limit: int = 180) -> str:
    if not value:
        return ""
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _comparison_rows(comparison, unavailable):
    baseline = comparison["modes"]["baseline"]
    corrected = comparison["modes"]["corrected"]

    rows = [
        {
            "Metric": "False-answer rate",
            "Baseline": _rate(baseline["false_answer_rate"]),
            "Corrected": _rate(corrected["false_answer_rate"]),
            "n": (
                f"Baseline {baseline['adversarial_n']}; "
                f"Corrected {corrected['adversarial_n']} adversarial runs"
            ),
        },
        {
            "Metric": "Correct-abstention rate",
            "Baseline": _rate(baseline["correct_abstention_rate"]),
            "Corrected": _rate(corrected["correct_abstention_rate"]),
            "n": (
                f"Baseline {baseline['adversarial_n']}; "
                f"Corrected {corrected['adversarial_n']} adversarial runs"
            ),
        },
        {
            "Metric": "General abstention rate",
            "Baseline": _rate(
                baseline["abstained_n"] / baseline["n_runs"]
                if baseline["n_runs"] else None
            ),
            "Corrected": _rate(
                corrected["abstained_n"] / corrected["n_runs"]
                if corrected["n_runs"] else None
            ),
            "n": (
                f"Baseline {baseline['n_runs']}; "
                f"Corrected {corrected['n_runs']} runs"
            ),
        },
        {
            "Metric": "Judge-approved after Retry",
            "Baseline": _rate(baseline["judge_approved_after_retry"]),
            "Corrected": _rate(corrected["judge_approved_after_retry"]),
            "n": (
                f"Baseline {baseline['judge_approved_after_retry_n']}; "
                f"Corrected {corrected['judge_approved_after_retry_n']} "
                "retry-eligible runs"
            ),
        },
    ]

    rows.extend(
        {
            "Metric": metric,
            "Baseline": value,
            "Corrected": value,
            "n": "—",
        }
        for metric, value in unavailable.items()
    )
    return rows


def _comparison_chart(comparison):
    chart_rows = []
    chart_metrics = (
        ("False-answer rate", "false_answer_rate"),
        ("Correct-abstention rate", "correct_abstention_rate"),
    )
    for key, label in MODES:
        for metric, field in chart_metrics:
            value = comparison["modes"][key][field]
            if value is not None:
                chart_rows.append(
                    {"metric": metric, "mode": label, "rate": value, "label": f"{value:.1%}"}
                )
    if not chart_rows:
        st.caption("No adversarial runs are recorded, so these rates cannot be charted.")
        return

    data = alt.Data(values=chart_rows)
    base = alt.Chart(data).encode(
        x=alt.X(
            "rate:Q",
            title=None,
            scale=alt.Scale(domain=[0, 1.12]),
            axis=alt.Axis(format=".0%", tickCount=5),
        ),
        y=alt.Y(
            "metric:N",
            title=None,
            sort=["False-answer rate", "Correct-abstention rate"],
        ),
        yOffset=alt.YOffset("mode:N"),
        color=alt.Color(
            "mode:N",
            title=None,
            scale=alt.Scale(
                domain=["Baseline", "Corrected"],
                range=[BASELINE_COLOR, CORRECTED_COLOR],
            ),
        ),
        tooltip=["metric:N", "mode:N", alt.Tooltip("rate:Q", format=".2%")],
    )
    bars = base.mark_bar(size=18)
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("label:N"))
    st.altair_chart((bars + labels).properties(height=150), width="stretch")


def _execution_section(connection, comparison):
    st.header("Execution and retry behavior")

    retry_rows = []
    for key, label in MODES:
        retry_percentage = comparison["modes"][key]["retry_percentage"]
        retry_rows.append(
            {
                "Mode": label,
                "Retry percentage": (
                    "n/a" if retry_percentage is None
                    else f"{retry_percentage:.2f}%"
                ),
                "Average retries": _number(queries.get_average_retries(connection, key)),
                "Recorded runs": comparison["modes"][key]["n_runs"],
            }
        )
    st.dataframe(retry_rows, hide_index=True, width="stretch")

    outcome_rows = [
        {"Mode": label, "Outcome": outcome, "Runs": count}
        for key, label in MODES
        for outcome, count in queries.get_final_outcome_split(connection, key).items()
    ]
    if outcome_rows:
        st.subheader("Final outcomes")
        st.dataframe(outcome_rows, hide_index=True, width="stretch")
    else:
        st.caption("No final outcomes are recorded.")

    failure_rows = [
        {"Mode": label, "Failure type": failure, "Attempts": count}
        for key, label in MODES
        for failure, count in queries.get_failure_type_breakdown(connection, key).items()
    ]
    st.subheader("Failure types")
    if failure_rows:
        st.dataframe(failure_rows, hide_index=True, width="stretch")
    else:
        st.caption("No judge failure types are recorded.")

    attempt_rows = [
        {
            "Mode": label,
            "Attempt number": item["attempt_number"],
            "Attempts": item["attempt_count"],
            "Average latency (ms)": item["average_latency_ms"],
            "Average cost": _cost(item["average_cost"]),
        }
        for key, label in MODES
        for item in queries.get_latency_cost_by_attempt_number(connection, key)
    ]
    st.subheader("Latency and cost by attempt")
    if attempt_rows:
        st.dataframe(attempt_rows, hide_index=True, width="stretch")
    else:
        st.caption("No attempts are recorded.")

    stage_rows = [
        {
            "Mode": label,
            "Stage": item["stage"],
            "Attempts": item["attempt_count"],
            "Average latency (ms)": item["average_latency_ms"],
            "Average cost": _cost(item["average_cost"]),
        }
        for key, label in MODES
        for item in queries.get_per_stage_latency_cost_breakdown(connection, key)
        if item["attempt_count"]
    ]
    st.subheader("Per-stage latency and cost")
    if stage_rows:
        st.dataframe(stage_rows, hide_index=True, width="stretch")
    else:
        st.caption("No attempt-level stage data is recorded.")


def _decision_trace_section(connection, run_summaries):
    st.header("Decision trace and failure cases")
    if not run_summaries:
        st.caption("No recorded runs are available for inspection.")
        return

    runs_by_id = {run["run_id"]: run for run in run_summaries}
    selected_id = st.selectbox(
        "Select a recorded run",
        list(runs_by_id),
        format_func=lambda run_id: (
            f"{runs_by_id[run_id]['mode'].title()} · "
            f"{runs_by_id[run_id]['original_query']} · {run_id[:8]}"
        ),
    )
    trace = queries.get_decision_trace(connection, selected_id)
    if trace is None:
        st.caption("The selected run no longer exists in the trace database.")
    else:
        st.markdown("**Original query**")
        st.write(trace["run"]["original_query"])
        st.markdown(f"**Final outcome:** {trace['run']['final_outcome']}")
        for position, attempt in enumerate(trace["attempts"], start=1):
            st.subheader(f"Attempt {position}")
            st.markdown("**Query used**")
            st.write(attempt["query_used"])
            st.markdown("**Judge reason**")
            st.write(attempt["judge_reason"] or "Not recorded")
            st.markdown("**Generated answer**")
            st.write(attempt["generated_answer"])
        if not trace["attempts"]:
            st.caption("This run has no attempt records.")

    st.subheader("Failure cases")
    st.caption(
        "Showing runs ending in judge rejection. SQLite does not store answer-correctness "
        "labels, so this list does not identify factual errors."
    )
    failures = [
        run for run in run_summaries
        if run["final_outcome"] == "abstained_judge_rejected"
    ]
    if not failures:
        st.caption("No judge-rejected runs are recorded.")
        return

    failure_rows = [
        {
            "Original query": _truncate(run["original_query"]),
            "Rewritten query": _truncate(run["query_used"]),
            "Answer": _truncate(run["generated_answer"]),
            "Outcome": run["final_outcome"],
        }
        for run in failures
    ]
    st.dataframe(failure_rows, hide_index=True, width="stretch")
    with st.expander("Inspect full failure-case text"):
        for run in failures:
            st.markdown(f"**Original query — {run['run_id']}**")
            st.write(run["original_query"])
            st.markdown("**Rewritten query**")
            st.write(run["query_used"] or "Not recorded")
            st.markdown("**Answer**")
            st.write(run["generated_answer"] or "Not recorded")
            st.markdown(f"**Outcome:** {run['final_outcome']}")


def main():
    st.set_page_config(page_title="RAG Evaluation and Observability", layout="wide")
    st.title("RAG Evaluation and Observability")
    st.write(
        "Recorded baseline and corrected runs, their outcomes, latency, retries, and decision history."
    )
    st.caption("Data source: `data/traces.db`. Read-only.")

    try:
        connection = queries.open_readonly_connection()
    except sqlite3.Error:
        st.caption("The trace database is unavailable or unreadable.")
        return

    try:
        total_runs = queries.get_total_runs(connection)
        comparison = queries.get_mode_comparison(connection)
        unavailable = queries.get_unavailable_evaluation_metrics()
        run_summaries = queries.get_run_summaries(connection)

        st.header("Baseline vs Corrected comparison")
        st.caption(comparison["population_label"])
        st.caption(f"Total recorded runs: {total_runs}")
        table_rows = _comparison_rows(comparison, unavailable)
        st.dataframe(table_rows, hide_index=True, width="stretch")

        st.header("False-answer and correct-abstention rates")
        st.caption("Both rates are calculated on the adversarial subset only.")
        _comparison_chart(comparison)

        _execution_section(connection, comparison)
        _decision_trace_section(connection, run_summaries)
    except sqlite3.Error:
        st.caption("The trace database is empty or does not match the expected schema.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()

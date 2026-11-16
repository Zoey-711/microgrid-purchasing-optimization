"""End-to-end solver for Question 1."""

from __future__ import annotations

import logging

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from src.config import (
    Q1_PROCESSED_FILE,
    Q1_SOLVE_LOG_FILE,
    Q1_SUBMISSION_CHECK_FILE,
    Q1_VALIDATION_LOG_FILE,
)
from src.data_io import load_q1_data
from src.export_results import (
    export_q1_baseline_interface,
    export_q1_paper_tables,
    export_q1_result,
)
from src.model_storage import build_q1_model, extract_q1_solution
from src.validation import (
    validate_q1,
    validate_q1_submission,
    write_submission_check,
    write_validation_report,
)
from src.visualization import plot_q1_results


def _logger() -> logging.Logger:
    Q1_SOLVE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("q1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(Q1_SOLVE_LOG_FILE, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _highs_solver():
    """Create an available HiGHS interface without falling back to another solver."""
    for solver_name in ("appsi_highs", "highs"):
        solver = pyo.SolverFactory(solver_name)
        if solver is not None and solver.available(exception_flag=False):
            return solver, solver_name
    raise RuntimeError("HiGHS is unavailable. Install highspy before solving Question 1.")


def solve_q1() -> dict:
    """Run read -> LP -> solve -> validate -> export -> plot for Question 1."""
    logger = _logger()
    data = load_q1_data()
    Q1_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(Q1_PROCESSED_FILE, index=False, encoding="utf-8-sig")
    logger.info("Loaded and converted %d periods from Attachment 1.", len(data))

    model = build_q1_model(data)
    solver, solver_name = _highs_solver()
    raw_result = solver.solve(model)
    status = f"{raw_result.solver.status}; {raw_result.solver.termination_condition}"
    logger.info("Solver interface: %s; status: %s", solver_name, status)
    if raw_result.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(f"Question 1 did not reach an optimal solution: {status}")

    objective_cost = float(pyo.value(model.objective))
    solution = extract_q1_solution(model, data)
    validation = validate_q1(solution, status, objective_cost)
    write_validation_report(validation, Q1_VALIDATION_LOG_FILE)
    logger.info("Validation passed: %s", validation["passed"])
    if not validation["passed"]:
        raise RuntimeError("Question-1 validation failed; result template was not written.")

    result_path = export_q1_result(solution)
    baseline_path = export_q1_baseline_interface(solution)
    paper_tables_path = export_q1_paper_tables(solution)
    submission_check = validate_q1_submission(
        solution, validation, result_path, paper_tables_path
    )
    write_submission_check(submission_check, Q1_SUBMISSION_CHECK_FILE)
    if not submission_check["passed"]:
        raise RuntimeError("Question-1 final submission check failed.")
    figure_paths = plot_q1_results(solution)
    logger.info("Exported result: %s", result_path)
    logger.info("Exported deterministic baseline interface: %s", baseline_path)
    logger.info("Final submission check passed: %s", submission_check["passed"])
    return {
        "objective_cost_yuan": objective_cost,
        "solution": solution,
        "validation": validation,
        "result_path": result_path,
        "baseline_path": baseline_path,
        "paper_tables_path": paper_tables_path,
        "submission_check": submission_check,
        "figure_paths": figure_paths,
    }

"""问题 3：多时点光伏预报下的滚动调整优化入口。"""


def run_q3_minimal_closure():
    """运行第二阶段三日因果闭环；不会生成 result3.xlsx。"""
    from src.validate_q3_small import run_q3_small_validation

    return run_q3_small_validation()


def solve_q3():
    """全年正式求解尚未授权，防止误生成正式结果。"""
    raise NotImplementedError("Question 3 full-year solve is not enabled; run_q3_minimal_closure only")

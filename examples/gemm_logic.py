"""GEMM 逻辑 Demo（硬件无关）。"""

import tasklang as task


@task.define(leaf=True)
def gemm_leaf_logic(C, A, B):
    """叶子级 GEMM 逻辑，占位实现。"""
    task.call_primitive("gemm", C, A, B)


@task.define
def gemm_block_logic(C, A, B):
    """块级 GEMM 逻辑，将大矩阵拆分为子块。"""
    Cp = task.partition(C, by="blocks", shape_from="A")
    Ap = task.partition(A, by="blocks", shape_from="A")
    Bp = task.partition(B, by="blocks", shape_from="A")

    # 并行遍历子块并启动叶子任务
    # 这里假设 Cp.shape = (M, N)
    for i, j in task.parallel_range((len(Cp), len(Cp[0]))):
        task.launch(gemm_leaf_logic, Cp[i][j], Ap[i], Bp[j]) 
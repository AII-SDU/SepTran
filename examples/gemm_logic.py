"""GEMM 逻辑 Demo（硬件无关）。

这个文件展示了 SepTran 的核心理念：用户编写纯粹的、硬件无关的算法逻辑。
编译器会根据映射文件自动生成高性能的 TileLang 代码。"""

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
    for i, j in task.parallel_range(Cp.shape):
        task.launch(gemm_leaf_logic, Cp[i, j], Ap[i], Bp[j]) 
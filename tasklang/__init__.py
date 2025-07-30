"""Task Lang 前端 API 占位实现。
仅用于 Demo 目的，后续应替换为真正的 TileLang/TVM Task 描述 DSL。"""

from __future__ import annotations

from functools import wraps
from itertools import product
from typing import Any, Callable, Iterable, Tuple

__all__ = [
    "define",
    "partition",
    "parallel_range",
    "launch",
    "call_primitive",
    "compile",
]


# -----------------------------------------------------------------------------
# Decorators & 前端语义占位实现
# -----------------------------------------------------------------------------

def define(func: Callable | None = None, *, leaf: bool = False):  # noqa: D401
    """声明一个 Task。

    Parameters
    ----------
    func : Callable
        被装饰的函数。
    leaf : bool, 默认 False
        是否为叶子任务。叶子任务不再递归分解。
    """

    def _decorator(f: Callable) -> Callable:
        f.__task_is_leaf__ = leaf  # type: ignore[attr-defined]

        @wraps(f)
        def _wrapper(*args, **kwargs):  # noqa: D401
            # TODO: 记录 Task 调用以生成 IR，目前直接执行函数主体
            return f(*args, **kwargs)

        return _wrapper

    # 装饰器既可以带参数，也可以不带参数
    if func is not None:
        return _decorator(func)
    return _decorator


# -----------------------------------------------------------------------------
# DSL Helper 占位实现
# -----------------------------------------------------------------------------

def partition(tensor: Any, *, by: str, shape_from: str | None = None) -> Any:  # noqa: D401
    """对张量进行逻辑分区。

    当前仅回传原对象作为占位。
    """

    # TODO: 返回一个表达逻辑分区的 IR 节点
    return tensor


def parallel_range(shape: Tuple[int, int] | Iterable[int]):  # noqa: D401
    """并行循环范围占位实现。"""

    if isinstance(shape, tuple):
        return product(range(shape[0]), range(shape[1]))
    return product(*[range(s) for s in shape])


def launch(task_fn: Callable, *args, **kwargs):  # noqa: D401
    """启动子任务占位实现。"""

    # TODO: 生成 Task 调用节点而非立即执行
    return task_fn(*args, **kwargs)


def call_primitive(name: str, *args, **kwargs):  # noqa: D401
    """调用底层原语的占位实现。"""

    # TODO: 与 TileLang / TVM 原语库对接
    print(f"[TaskLang] TODO: call primitive '{name}' with args={args}, kwargs={kwargs}") 

# -----------------------------------------------------------------------------
# Engine API
# -----------------------------------------------------------------------------

from .engine import compile  # noqa: F401 
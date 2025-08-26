"""Task Lang 前端 API - 基于 TileLang 的硬件无关算法描述语言。

核心思想：用户编写硬件无关的算法逻辑，编译器根据映射文件自动生成 TileLang 代码。"""

from __future__ import annotations

from functools import wraps
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Import TileLang for later code generation
try:
    import tilelang
    import tilelang.language as T
    TILELANG_AVAILABLE = True
except ImportError:
    TILELANG_AVAILABLE = False
    print("Warning: TileLang not found. Please install TileLang for full functionality.")

__all__ = [
    "define",
    "partition", 
    "parallel_range",
    "launch",
    "call_primitive",
    "compile",
    "get_task_registry",
    "reset_task_registry",
    "TaskDef",
    "TensorPartition",
    "PrimitiveCall",
]


# -----------------------------------------------------------------------------
# Task 定义和调用图构建
# -----------------------------------------------------------------------------

# 全局任务注册表
_TASK_REGISTRY: Dict[str, 'TaskDef'] = {}

class TaskDef:
    """任务定义，包含函数和元数据。"""
    
    def __init__(self, func: Callable, is_leaf: bool = False):
        self.func = func
        self.is_leaf = is_leaf
        self.name = func.__name__
        self.calls = []  # 记录此任务调用的其他任务
    
    def add_call(self, callee_name: str, args, kwargs):
        """记录任务调用。"""
        self.calls.append((callee_name, args, kwargs))

# -----------------------------------------------------------------------------
# Task DSL 前端实现
# -----------------------------------------------------------------------------

def define(func: Optional[Callable] = None, *, leaf: bool = False):  # noqa: D401
    """声明一个 Task。

    Parameters
    ----------
    func : Callable
        被装饰的函数。
    leaf : bool, 默认 False
        是否为叶子任务。叶子任务不再递归分解。
    """

    def _decorator(f: Callable) -> Callable:
        # 创建任务定义并注册
        task_def = TaskDef(f, is_leaf=leaf)
        _TASK_REGISTRY[f.__name__] = task_def
        
        f.__task_def__ = task_def  # type: ignore[attr-defined]
        f.__task_is_leaf__ = leaf  # type: ignore[attr-defined]

        @wraps(f)
        def _wrapper(*args, **kwargs):  # noqa: D401
            # 在编译时，我们会分析调用图而不是执行函数
            # 但为了保持兼容性，这里仍然执行原函数
            return f(*args, **kwargs)

        return _wrapper

    # 装饰器既可以带参数，也可以不带参数
    if func is not None:
        return _decorator(func)
    return _decorator


# -----------------------------------------------------------------------------
# DSL Helper 实现 - 构建中间表示
# -----------------------------------------------------------------------------

class TensorPartition:
    """张量分区的中间表示。"""
    
    def __init__(self, tensor: Any, partition_type: str, shape_from: Optional[str] = None):
        self.tensor = tensor
        self.partition_type = partition_type
        self.shape_from = shape_from
        # 假设分区后的形状，实际应根据映射文件确定
        self.shape = (2, 2)  # 占位
    
    def __getitem__(self, index):
        return TensorSlice(self, index)
    
    def __len__(self):
        return self.shape[0]

class TensorSlice:
    """张量切片的中间表示。"""
    
    def __init__(self, partition: TensorPartition, index):
        self.partition = partition
        self.index = index

def partition(tensor: Any, *, by: str, shape_from: Optional[str] = None) -> TensorPartition:  # noqa: D401
    """对张量进行逻辑分区。
    
    Args:
        tensor: 要分区的张量
        by: 分区方式 ("blocks" 等)
        shape_from: 从哪个张量推断分区形状
    
    Returns:
        TensorPartition: 分区后的张量表示
    """
    return TensorPartition(tensor, by, shape_from)


def parallel_range(shape) -> Iterable[Tuple[int, ...]]:  # noqa: D401
    """并行循环范围占位实现。"""

    if isinstance(shape, tuple):
        return product(range(shape[0]), range(shape[1]))
    return product(*[range(s) for s in shape])


def launch(task_fn: Callable, *args, **kwargs):  # noqa: D401
    """启动子任务，记录调用关系用于后续代码生成。"""
    
    # 记录任务调用关系（用于构建调用图）
    caller_task = _get_current_task_context()
    if caller_task and hasattr(task_fn, '__task_def__'):
        caller_task.add_call(task_fn.__name__, args, kwargs)
    
    # 为了兼容性，仍然执行原函数
    return task_fn(*args, **kwargs)

def _get_current_task_context() -> Optional[TaskDef]:
    """获取当前任务上下文（简化实现）。"""
    # 这里需要实现调用栈追踪，暂时返回None
    return None


def call_primitive(name: str, *args, **kwargs):  # noqa: D401
    """调用底层原语，将被映射到具体的 TileLang 原语。"""
    
    # 记录原语调用，稍后会根据映射文件转换为具体的 TileLang 调用
    print(f"[TaskLang] Primitive call: {name}(*{args}, **{kwargs})")
    return PrimitiveCall(name, args, kwargs)

class PrimitiveCall:
    """原语调用的中间表示。"""
    
    def __init__(self, name: str, args: Tuple, kwargs: Dict):
        self.name = name
        self.args = args
        self.kwargs = kwargs

# -----------------------------------------------------------------------------
# 调用图分析和代码生成支持
# -----------------------------------------------------------------------------

def get_task_registry() -> Dict[str, TaskDef]:
    """获取全局任务注册表。"""
    return _TASK_REGISTRY.copy()

def reset_task_registry():
    """重置任务注册表（用于测试）。"""
    global _TASK_REGISTRY
    _TASK_REGISTRY.clear() 

# -----------------------------------------------------------------------------
# Engine API
# -----------------------------------------------------------------------------

from .engine import compile  # noqa: F401 
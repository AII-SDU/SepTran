"""SepTran 代码生成器 - 将逻辑任务 + 映射文件转换为 TileLang 代码。

这是 SepTran 的核心组件，实现"逻辑映射分离"的关键转换逻辑。
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Set, Tuple

from . import TaskDef, get_task_registry
from .mapping import MappingInfo


class CodeGenerator:
    """SepTran 代码生成器。
    
    根据用户编写的逻辑任务和映射文件，生成对应的 TileLang 代码。
    """
    
    def __init__(self, mapping_info: MappingInfo):
        self.mapping_info = mapping_info
        self.task_registry = get_task_registry()
        self.generated_functions: Dict[str, str] = {}
    
    def generate_tilelang_code(self, entry_task_name: str) -> str:
        """生成完整的 TileLang 代码。
        
        Args:
            entry_task_name: 入口任务名称
            
        Returns:
            str: 生成的 TileLang 代码
        """
        # 构建依赖图
        dependencies = self._build_dependency_graph(entry_task_name)
        
        # 按拓扑顺序生成函数
        code_parts = [
            "import tilelang",
            "import tilelang.language as T",
            "",
            "# SepTran 自动生成的 TileLang 代码",
            "# 基于逻辑任务和映射文件自动转换",
            "",
        ]
        
        # 生成所有逻辑任务的 TileLang 函数
        for task_name in self.task_registry.keys():
            function_code = self._generate_task_function(task_name)
            code_parts.append(function_code)
            code_parts.append("")
        
        # 生成主编译函数（确保入口任务也被处理）
        if entry_task_name not in dependencies:
            # 如果入口任务不在逻辑任务中，需要生成映射任务
            print(f"[CodeGenerator] 入口任务 {entry_task_name} 不在依赖图中，生成映射任务函数")
            mapping_function = self._generate_mapping_task_function(entry_task_name)
            code_parts.append(mapping_function)
            code_parts.append("")
        else:
            print(f"[CodeGenerator] 入口任务 {entry_task_name} 在依赖图中")
        
        main_function = self._generate_main_compile_function(entry_task_name)
        code_parts.append(main_function)
        
        generated_code = "\n".join(code_parts)
        print(f"[CodeGenerator] 依赖图: {dependencies}")
        print(f"[CodeGenerator] 入口任务: {entry_task_name}")
        return generated_code
    
    def _build_dependency_graph(self, entry_task: str) -> List[str]:
        """构建任务依赖图，返回拓扑排序后的任务列表。"""
        visited: Set[str] = set()
        result: List[str] = []
        
        def dfs(task_name: str):
            if task_name in visited:
                return
            visited.add(task_name)
            
            # 分析任务的调用关系
            task_def = self.task_registry.get(task_name)
            if task_def:
                # 分析函数源码找出调用的其他任务
                calls = self._analyze_task_calls(task_def)
                for called_task in calls:
                    dfs(called_task)
                result.append(task_name)
            else:
                # 任务不在逻辑代码中，可能是映射任务
                print(f"[CodeGenerator] 任务 {task_name} 不在逻辑代码中，可能是映射任务")
        
        dfs(entry_task)
        return result
    
    def _analyze_task_calls(self, task_def: TaskDef) -> List[str]:
        """分析任务中调用的其他任务。"""
        # 简化实现：通过源码分析找出 launch 调用
        source = inspect.getsource(task_def.func)
        calls = []
        
        # 查找 task.launch(func_name, ...) 模式
        lines = source.split('\n')
        for line in lines:
            line = line.strip()
            if 'task.launch(' in line or 'launch(' in line:
                # 简单的模式匹配，实际应该用更复杂的 AST 分析
                for task_name in self.task_registry:
                    if task_name in line and task_name != task_def.name:
                        calls.append(task_name)
                        break
        
        return calls
    
    def _generate_task_function(self, task_name: str) -> str:
        """为特定任务生成 TileLang 函数。"""
        task_def = self.task_registry[task_name]
        mapping = self.mapping_info.get_task_mapping(task_name)
        
        if task_def.is_leaf:
            return self._generate_leaf_task(task_name, task_def, mapping)
        else:
            return self._generate_composite_task(task_name, task_def, mapping)
    
    def _generate_leaf_task(self, task_name: str, task_def: TaskDef, mapping: Optional[Dict]) -> str:
        """生成叶子任务的 TileLang 代码。"""
        if not mapping:
            # 使用默认映射
            return f"""# 叶子任务: {task_name} (使用默认映射)
def {task_name}_tilelang(A, B, C):
    # TODO: 实现具体的 TileLang 原语调用
    T.gemm(A, B, C)  # 占位实现"""
        
        # 根据映射文件中的 leaf_task_bindings 生成代码
        leaf_bindings = mapping.get('leaf_task_bindings', {})
        primitive_name = leaf_bindings.get(task_name, 'gemm')  # 默认为 gemm
        
        return f"""# 叶子任务: {task_name} -> {primitive_name}
def {task_name}_tilelang(A, B, C):
    # 映射到 TileLang 原语: {primitive_name}
    T.{primitive_name}(A, B, C)"""
    
    def _generate_composite_task(self, task_name: str, task_def: TaskDef, mapping: Optional[Dict]) -> str:
        """生成复合任务的 TileLang 代码。"""
        if not mapping:
            return f"""# 复合任务: {task_name} (无映射配置)
def {task_name}_tilelang(*args, **kwargs):
    # 需要映射文件配置
    pass"""
        
        # 获取内存配置
        mems = mapping.get('mems', {})
        proc = mapping.get('proc', 'BLOCK')
        pipeline = mapping.get('pipeline')
        
        # 生成内存分配代码
        memory_allocs = []
        for var_name, mem_scope in mems.items():
            if mem_scope == 'SHARED':
                memory_allocs.append(f"    {var_name}_shared = T.alloc_shared((block_M, block_K), dtype)")
            elif mem_scope == 'REGISTER':
                memory_allocs.append(f"    {var_name}_local = T.alloc_fragment((block_M, block_N), accum_dtype)")
        
        memory_code = "\n".join(memory_allocs) if memory_allocs else "    # 无特殊内存分配"
        
        # 生成流水线代码
        pipeline_code = ""
        if pipeline:
            pipeline_code = f"    # 启用 {pipeline} 级流水线\n    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages={pipeline}):"
        else:
            pipeline_code = "    for k in range(T.ceildiv(K, block_K)):"
        
        return f"""# 复合任务: {task_name} (映射到 {proc})
@T.prim_func  
def {task_name}_tilelang(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
{memory_code}
        
        T.clear(C_local)
{pipeline_code}
            # 根据映射自动插入数据移动
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            
            # 调用子任务（将被映射到具体实现）
            # {task_name} -> 子任务调用
            
        T.copy(C_local, C[by * block_M, bx * block_N])"""
    
    def _generate_mapping_task_function(self, mapping_task_name: str) -> str:
        """为映射中的任务生成函数（在逻辑中不存在的任务）。"""
        mapping = self.mapping_info.get_task_mapping(mapping_task_name)
        if not mapping:
            return f"""# 映射任务: {mapping_task_name} (无配置)
def {mapping_task_name}_tilelang(*args, **kwargs):
    pass"""
        
        task_name = mapping.get('task_name')
        if task_name and task_name in self.task_registry:
            # 映射到已存在的逻辑任务
            return f"""# 映射任务: {mapping_task_name} -> {task_name}
def {mapping_task_name}_tilelang(*args, **kwargs):
    return {task_name}_tilelang(*args, **kwargs)"""
        else:
            # 生成默认的 TileLang 实现
            return self._generate_default_tilelang_function(mapping_task_name, mapping)
    
    def _generate_default_tilelang_function(self, task_name: str, mapping: Dict) -> str:
        """为没有逻辑对应的映射任务生成默认的 TileLang 函数。"""
        mems = mapping.get('mems', {})
        proc = mapping.get('proc', 'BLOCK')
        pipeline = mapping.get('pipeline')
        
        return f"""# 映射任务: {task_name} (根据映射配置生成)
@T.prim_func
def {task_name}_tilelang(
    A: T.Tensor(("M", "K"), "float16"),
    B: T.Tensor(("K", "N"), "float16"), 
    C: T.Tensor(("M", "N"), "float16"),
):
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32
    
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), "float16")
        B_shared = T.alloc_shared((block_K, block_N), "float16")
        C_local = T.alloc_fragment((block_M, block_N), "float")
        
        T.clear(C_local)
        
        {'for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=' + str(pipeline) + '):' if pipeline else 'for k in range(T.ceildiv(K, block_K)):'}
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared) 
            T.gemm(A_shared, B_shared, C_local)
            
        T.copy(C_local, C[by * block_M, bx * block_N])"""
    
    def _generate_main_compile_function(self, entry_task: str) -> str:
        """生成主编译函数。"""
        return f"""# 主编译函数
@tilelang.jit(out_idx=[-1])  
def septran_compiled_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    return {entry_task}_tilelang

def compile_septran_kernel(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32):
    \"\"\"编译 SepTran 内核。\"\"\"
    kernel = septran_compiled_kernel(M, N, K, block_M, block_N, block_K)
    return kernel"""
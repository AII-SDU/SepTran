"""SepTran Engine：解析 Task Lang + Mapping YAML → TileLang 代码。

实现“逻辑映射分离”的核心编译流程。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tempfile
from pathlib import Path

from .mapping import load_yaml, MappingInfo
from .codegen import CodeGenerator

# 尝试导入 TileLang
try:
    import tilelang
    TILELANG_AVAILABLE = True
except ImportError:
    TILELANG_AVAILABLE = False

__all__ = ["compile"]


class SepTranEngine:
    """负责 orchestrate 前端解析、映射融合、TileLang 代码生成。"""

    def __init__(self, logic_file: Path, mapping_file: Path):
        self.logic_file = Path(logic_file)
        self.mapping_file = Path(mapping_file)
        self.mapping_info = load_yaml(self.mapping_file)
        self.codegen = CodeGenerator(self.mapping_info)

    # ---------------------------------------------------------------------
    # 高层 API
    # ---------------------------------------------------------------------

    def build(self) -> Any:  # noqa: D401
        """执行完整编译流程。"""
        print(f"[SepTranEngine] 开始编译 {self.logic_file} + {self.mapping_file}")
        
        # 1. 加载逻辑文件（执行 Python 代码以注册任务）
        self._load_logic_file()
        
        # 2. 获取入口点任务
        entry_task = self.mapping_info.get_entrypoint()
        if not entry_task:
            raise ValueError("映射文件中未指定 entrypoint")
        
        print(f"[SepTranEngine] 入口点任务: {entry_task}")
        
        # 3. 生成 TileLang 代码
        tilelang_code = self.codegen.generate_tilelang_code(entry_task)
        
        print(f"[SepTranEngine] 已生成 TileLang 代码（{len(tilelang_code)} 字符）")
        print("=== 生成的 TileLang 代码 ===")
        print(tilelang_code)
        print("=== 代码结束 ===")
        
        if not TILELANG_AVAILABLE:
            print("[SepTranEngine] 警告: TileLang 未安装，返回生成的代码字符串")
            return {
                'generated_code': tilelang_code,
                'status': 'code_only',
                'message': 'TileLang not available - returning generated code only'
            }
        
        # 4. 编译 TileLang 代码为可执行内核
        kernel = self._compile_tilelang_code(tilelang_code)
        
        print("[SepTranEngine] 编译完成")
        return kernel
    
    def _load_logic_file(self):
        """加载逻辑文件，注册任务定义。"""
        # 清空任务注册表
        from . import reset_task_registry
        reset_task_registry()
        
        # 导入 tasklang 模块以供逻辑文件使用
        import tasklang as task
        namespace = {
            '__name__': '__main__',
            'task': task,
            'tasklang': task  # 兼容两种导入方式
        }
        
        # 执行 Python 文件
        exec(self.logic_file.read_text(encoding='utf-8'), namespace)
        
        # 更新 codegen 中的任务注册表
        from . import get_task_registry
        self.codegen.task_registry = get_task_registry()
        
        print(f"[SepTranEngine] 已加载逻辑文件，注册了 {len(self.codegen.task_registry)} 个任务")
        if self.codegen.task_registry:
            print(f"[SepTranEngine] 注册的任务: {list(self.codegen.task_registry.keys())}")
    
    def _compile_tilelang_code(self, tilelang_code: str) -> Any:
        """编译 TileLang 代码为可执行内核。"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(tilelang_code)
            temp_file = f.name
        
        try:
            # 执行生成的 TileLang 代码
            namespace = {'__name__': '__main__'}
            exec(Path(temp_file).read_text(), namespace)
            
            # 获取编译函数  
            if 'compile_septran_kernel' in namespace:
                return {
                    'kernel': namespace['compile_septran_kernel'](),
                    'generated_code': tilelang_code,
                    'status': 'success'
                }
            else:
                return {
                    'generated_code': tilelang_code,
                    'status': 'error', 
                    'message': '生成的代码中没有找到 compile_septran_kernel 函数'
                }
        finally:
            # 清理临时文件
            Path(temp_file).unlink(missing_ok=True)


def compile(*, logic_file, mapping_file) -> Any:  # noqa: D401
    """编译入口函数，封装 ``SepTranEngine``。"""
    engine = SepTranEngine(Path(logic_file), Path(mapping_file))
    return engine.build() 
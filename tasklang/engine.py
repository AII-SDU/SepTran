"""SepTran Engine：解析 Task Lang + Mapping YAML → TensorIR（骨架）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import mapping as _mapping

__all__ = ["compile"]


class SepTranEngine:
    """负责 orchestrate 前端解析、映射融合、Pass pipeline 与 CodeGen。"""

    def __init__(self, logic_file: Path, mapping_file: Path):
        self.logic_file = Path(logic_file)
        self.mapping_file = Path(mapping_file)
        # 占位：真实实现会解析 Python 源码 → 初始 TIR
        self.logic_ir = None
        self.mapping_info = _mapping.load_yaml(self.mapping_file)

    # ---------------------------------------------------------------------
    # 高层 API
    # ---------------------------------------------------------------------

    def build(self) -> Any:  # noqa: D401
        """执行完整编译流程，当前仅打印调试信息。"""
        print("[Engine] TODO: 解析逻辑文件", self.logic_file)
        print("[Engine] TODO: 解析映射文件", self.mapping_file)
        print("[Engine] Mapping Info:", self.mapping_info)
        print("[Engine] TODO: 生成 TensorIR & 执行 Pass pipeline & CodeGen")
        return None


def compile(*, logic_file: str | Path, mapping_file: str | Path) -> Any:  # noqa: D401
    """编译入口函数，封装 ``SepTranEngine``。"""
    engine = SepTranEngine(Path(logic_file), Path(mapping_file))
    return engine.build() 
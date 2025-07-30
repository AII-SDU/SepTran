"""Mapping YAML 解析工具（骨架）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = ["load_yaml"]


def load_yaml(path: str | Path) -> Dict[str, Any]:  # noqa: D401
    """加载并解析 Mapping YAML。

    Parameters
    ----------
    path : str | Path
        YAML 文件路径。

    Returns
    -------
    Dict[str, Any]
        简单地返回 YAML 内容，后续可转换为数据类。
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data 
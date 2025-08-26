"""Mapping YAML 解析工具 - 提供结构化的映射信息访问。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

__all__ = ["load_yaml", "MappingInfo"]


class MappingInfo:
    """结构化的映射信息类，提供方便的访问接口。"""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.entrypoint = data.get('entrypoint')
        self.task_mappings = data.get('task_mappings', {})
    
    def get_entrypoint(self) -> Optional[str]:
        """获取入口点任务名。"""
        return self.entrypoint
    
    def get_task_mapping(self, task_name: str) -> Optional[Dict[str, Any]]:
        """获取指定任务的映射信息。"""
        return self.task_mappings.get(task_name)
    
    def get_all_task_names(self) -> List[str]:
        """获取所有任务名称。"""
        return list(self.task_mappings.keys())
    
    def get_memory_mapping(self, task_name: str) -> Dict[str, str]:
        """获取任务的内存映射配置。"""
        mapping = self.get_task_mapping(task_name)
        return mapping.get('mems', {}) if mapping else {}
    
    def get_processor_type(self, task_name: str) -> str:
        """获取任务的处理器类型。"""
        mapping = self.get_task_mapping(task_name)
        return mapping.get('proc', 'BLOCK') if mapping else 'BLOCK'
    
    def get_pipeline_stages(self, task_name: str) -> Optional[int]:
        """获取任务的流水线级数。"""
        mapping = self.get_task_mapping(task_name)
        return mapping.get('pipeline') if mapping else None
    
    def get_leaf_task_bindings(self, task_name: str) -> Dict[str, str]:
        """获取叶子任务绑定。"""
        mapping = self.get_task_mapping(task_name)
        return mapping.get('leaf_task_bindings', {}) if mapping else {}
    
    def get_tunables(self, task_name: str) -> Dict[str, Any]:
        """获取可调参数。"""
        mapping = self.get_task_mapping(task_name)
        return mapping.get('tunables', {}) if mapping else {}

def load_yaml(path: Union[str, Path]) -> MappingInfo:  # noqa: D401
    """加载并解析 Mapping YAML。

    Parameters
    ----------
    path : Union[str, Path]
        YAML 文件路径。

    Returns
    -------
    MappingInfo
        结构化的映射信息对象。
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return MappingInfo(data) 
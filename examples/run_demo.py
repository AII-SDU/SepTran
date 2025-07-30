"""运行 GEMM Demo。

当前仅调用占位 compile 接口，验证骨架完整性。"""

from pathlib import Path
from tasklang import compile


if __name__ == "__main__":
    root = Path(__file__).parent
    logic = root / "gemm_logic.py"
    mapping = root / "gemm_on_nvidia_gpu.mapping.yaml"

    binary = compile(logic_file=logic, mapping_file=mapping)

    print("[Demo] 编译完成，返回值:", binary) 
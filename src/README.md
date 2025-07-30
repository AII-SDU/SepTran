# src 目录说明

该目录用于存放 **C++/CUDA 等后端源码**，包括：

1. `transform/` — 编译 Pass（Mapping-Guided Lowering 等）
2. `target/`     — 各硬件 CodeGen（如 CodeGenDSA、CodeGenCPU）
3. `runtime/`    — 运行时封装（可选）

目前为空，后续实现时逐步补充。 
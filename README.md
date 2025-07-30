# SepTran：基于逻辑映射分离的AI编译器框架

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TVM](https://img.shields.io/badge/Based%20on-TVM-orange)](https://tvm.apache.org/)
[![Status](https://img.shields.io/badge/Status-Planning-yellow.svg)](https://github.com/AII-SDU/septran)

**🚧 项目规划中 🚧**  
**一个革命性的AI编译器框架，彻底分离算法逻辑与硬件映射，一次编写，多端部署**

</div>

## 🚀 项目愿景

SepTran 站在巨人的肩膀上，深度借鉴了 [TileLang](https://github.com/tile-ai/tilelang) 的解耦思想和 TVM TensorIR 的设计理念，同时创新性地引入了 [Cypress 论文](https://rohany.github.io/publications/pldi2025-cypress.pdf) "逻辑与映射分离" 的核心思想，构建了一个全新的AI编译器框架。

我们的目标是**彻底解决AI编译器的N×M复杂度问题**：让开发者为N个算子逻辑编写一次代码，通过M个硬件映射文件即可适配所有目标硬件，从根本上解决AI编译器的可移植性挑战。

### 🎯 核心价值
- **🔄 关注点分离**：算法逻辑与硬件优化完全解耦
- **📈 线性扩展性**：从N×M复杂度降低到N+M
- **⚡ 开发效率**：一次编写，多端部署
- **🎛️ 专家友好**：硬件专家可独立优化映射文件

## 🏗️ 架构设计

SepTran 采用创新的三层架构设计，基于 [TVM](https://tvm.apache.org/) 生态构建：

```
┌─────────────────────────────────────────────────────────────┐
│                    逻辑层 (Task Lang前端)                    │
│    ✨ 纯粹的、层次化的计算任务描述，完全硬件无关            │
└─────────────────────────────────────────────────────────────┘
                              ⬇️
┌─────────────────────────────────────────────────────────────┐
│                  映射层 (硬件映射文件)                      │
│    🎯 声明式的硬件映射规范，专家编写的性能优化知识          │
└─────────────────────────────────────────────────────────────┘
                              ⬇️
┌─────────────────────────────────────────────────────────────┐
│               编译器 (智能的映射处理器)                     │
│    🧠 映射指导下的代码生成，自动融合逻辑与映射              │
└─────────────────────────────────────────────────────────────┘
                              ⬇️
┌─────────────────────────────────────────────────────────────┐
│               TVM IR + 自定义 CodeGen                       │
│    ⚙️ 生成包含硬件库调用的 TVM TensorIR，转译为 C 库调用   │
└─────────────────────────────────────────────────────────────┘
```

## 💻 技术设计

### 第一步：编写硬件无关的逻辑代码

```python
import task_lang as task

# 叶子任务：纯粹的逻辑描述，无硬件相关代码
@task.define(leaf=True)
def gemm_leaf_logic(C, A, B):
    """GEMM叶子任务的逻辑定义"""
    task.call_primitive("gemm", C, A, B)

# 复合任务：层次化的算法分解
@task.define
def gemm_block_logic(C, A, B):
    """GEMM块级任务的逻辑分解"""
    # 纯粹的逻辑分解：将大问题分解成小问题
    Cp = task.partition(C, by="blocks", shape_from="A")
    Ap = task.partition(A, by="blocks", shape_from="A") 
    Bp = task.partition(B, by="blocks", shape_from="A")
    
    # 启动下一层级的任务
    for i, j in task.parallel_range(Cp.shape):
        task.launch(gemm_leaf_logic, Cp[i,j], Ap[i,:], Bp[:,j])
```

### 第二步：编写硬件映射文件

```yaml
# gemm_on_nvidia_gpu.mapping.yaml
entrypoint: gemm_host

task_mappings:
  gemm_host:
    task_name: gemm_host_logic
    proc: HOST
    mems: {C: GLOBAL, A: GLOBAL, B: GLOBAL}
    tunables: {block_size: 128}
    calls: {gemm_block_logic: gemm_block_gpu}

  gemm_block_gpu:
    task_name: gemm_block_logic
    proc: BLOCK
    # 核心指令：A和B放到SHARED内存，启用流水线
    mems: {C: REGISTER, A: SHARED, B: SHARED}
    pipeline: 3
    leaf_task_bindings:
      gemm_leaf_logic: gemm_leaf_tensorcore
```

### 第三步：智能编译过程

#### 映射指导下的 TVM TensorIR 生成

编译器从入口点开始，递归遍历逻辑任务树，在每个 `task.launch(callee, ...)` 节点：

1. **查询映射文件**：确定被调用任务应该使用哪个映射实例
2. **分析内存需求**：读取映射实例的 `mems` 字段，了解内存要求
3. **对比与决策**：发现内存空间不匹配时自动决策
4. **自动插入原语**：自动在生成的 IR 中插入必要的操作

**生成的 TVM TensorIR 示例：**

```python
@T.prim_func
def generated_kernel(
    A: T.handle, B: T.handle, C: T.handle,
    a_scratch: T.handle, b_scratch: T.handle
):
    A_buffer = T.match_buffer(...)
    B_buffer = T.match_buffer(...)
    C_buffer = T.match_buffer(...)
    A_scratch_buffer = T.match_buffer(...)
    B_scratch_buffer = T.match_buffer(...)

    # 编译器根据映射文件自动插入的 DMA 操作
    T.evaluate(T.call_extern(
        "dsa.dma_copy", A_buffer, A_scratch_buffer, dtype="void"
    ))
    T.evaluate(T.call_extern(
        "dsa.dma_copy", B_buffer, B_scratch_buffer, dtype="void"
    ))

    # 自动插入的同步操作
    T.evaluate(T.call_extern("dsa.sync", dtype="void"))

    # 核心计算
    T.evaluate(T.call_extern(
        "dsa.mma", C_buffer, A_scratch_buffer, B_scratch_buffer, dtype="void"
    ))
```

#### 自定义 TVM CodeGen

继承 TVM 现有的 `CodeGenC`，创建自定义的 `CodeGenDSA` 类：

```cpp
class CodeGenDSA : public CodeGenC {
public:
    void VisitStmt_(const CallNode* op) override {
        if (op->op.same_as(StringImm("dsa.dma_copy"))) {
            // 翻译为厂商 C 库的函数调用
            this->stream << "vendor_lib_dma_copy("
                       << this->GetVarName(op->args[0]) << ", "
                       << this->GetVarName(op->args[1]) << ");\n";
        } else if (op->op.same_as(StringImm("dsa.mma"))) {
            this->stream << "vendor_lib_mma("
                       << this->GetVarName(op->args[0]) << ", "
                       << this->GetVarName(op->args[1]) << ", "
                       << this->GetVarName(op->args[2]) << ");\n";
        } else {
            CodeGenC::VisitStmt_(op);
        }
    }
};
```

### 第四步：编译和执行

```python
import septran_compiler as stc

# 编译：逻辑 + 映射 → 高性能代码
kernel = stc.compile(
    logic_file="gemm_logic.py",
    mapping_file="gemm_on_nvidia_gpu.mapping.yaml"
)

# 执行
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = kernel(a, b)
```

## 🎯 设计目标：多硬件后端支持

我们的框架设计目标是通过编写相应的映射文件和 CodeGen 扩展，支持各种硬件平台：

- **NVIDIA GPUs**
- **AMD GPUs**:
- **DSA硬件**
- **CPU**

### 核心设计理念

**关注点分离**：TileLang将逻辑和策略**混合**在Python代码中，我们将其**彻底分离**，算法逻辑与硬件优化完全解耦，开发者只需：
- 编写一次**硬件无关的算法逻辑**（`.py` 文件）
- 为每个目标硬件编写**轻量级的映射文件**（`.yaml` 文件）

**线性扩展性**：N个算法 × M个硬件 = N+M个文件（而非N×M个代码实现）

**专家协作**：算法专家专注逻辑实现，硬件专家专注性能映射，各司其职

## 🔮 未来展望：自动优化探索

### 从手动映射到智能编译

第一阶段构建坚固的手动框架，第二阶段目标是实现**自动化的性能调优系统**：

```yaml
# 可搜索的映射文件示例
tunables:
  block_size_M: !tune {type: choice, values: [64, 128, 256]}
  pipeline_depth: !tune {type: choice, values: [2, 3, 4, 5]}
leaf_task_bindings:
  gemm_leaf_logic: !tune {type: choice, values: [gemm_leaf_tensorcore, gemm_leaf_simt]}
```

通过这个渐进式的研究路径，我们将实现从手动的高性能框架到真正意义上的智能编译系统的蜕变。

## 🤝 参与项目

> **🌟 寻找早期贡献者**：项目正在启动阶段，这是参与开源项目早期建设的绝佳机会！

我们欢迎社区参与！您可以通过以下方式加入：

- **💡 设计讨论**: 在 Issues 中分享对架构设计的想法和建议
- **📋 需求收集**: 告诉我们您希望支持的硬件和应用场景
- **👀 关注进展**: Star 本项目，获取最新开发动态
- **📝 早期贡献**: 等项目启动后参与代码开发
- **🎯 硬件专业知识**: 如果您是某个硬件平台的专家，欢迎分享映射文件的设计建议

## 📄 许可证

本项目采用 [MIT License](./LICENSE) 开源许可证。

## 🙏 致谢

- **[TVM社区](https://tvm.apache.org/)** - 提供了强大的编译器基础设施
- **[Cypress 论文](https://rohany.github.io/publications/pldi2025-cypress.pdf)** - 启发了"逻辑映射分离"的核心思想
- **[TileLang](https://github.com/tile-ai/tilelang)** - 提供了宝贵的设计参考和灵感

---

<div align="center">

**⭐ 如果您对这个项目感兴趣，请给我们一个Star关注进展！⭐**

[💬 参与讨论](https://github.com/AII-SDU/septran/issues) • 
[📋 提需求建议](https://github.com/AII-SDU/septran/issues)

**🚀 项目启动时我们会在这里公告，欢迎届时参与贡献！**

</div> 

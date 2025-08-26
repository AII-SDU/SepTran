# SepTran：基于逻辑映射分离的AI编译器框架

## 核心构想

**创建一个新的AI编译器框架，它借鉴Cypress"逻辑与映射分离"的核心思想，允许开发者为N个算子逻辑，手动编写M个硬件的"映射文件"。编译器的核心任务是读取这两部分，并智能地将它们组合，生成高性能的底层代码。**

这个目标清晰、务实，且直指N x M难题的解决方案：通过彻底分离算法逻辑与硬件映射，让一份算子代码可以通过不同的映射文件适配多个硬件后端，从根本上解决AI编译器的可移植性问题。

## 架构设计

### 第一层：逻辑层 - Task Lang前端

这一层完全模仿Cypress的`Logical Description`，负责描述"做什么"，提供纯粹的、层次化的计算任务描述。

#### 分层的任务定义
```python
import task_lang as task

# 装饰器，定义一个任务及其签名（读写权限等）
@task.define(leaf=True)
def gemm_leaf_logic(C, A, B):
    # 叶子任务的实现，将委托给TileLang/TVM的高性能原语
    # 注意：这里只是一个逻辑占位符，不涉及具体的TileLang调用
    task.call_primitive("gemm", C, A, B)

@task.define
def gemm_block_logic(C, A, B):
    # 纯粹的逻辑分解：将大问题分解成小问题
    Cp = task.partition(C, by="blocks", shape_from="A")
    Ap = task.partition(A, by="blocks", shape_from="A")
    Bp = task.partition(B, by="blocks", shape_from="A")

    # 启动下一层级的任务
    for i, j in task.parallel_range(Cp.shape):
        task.launch(gemm_leaf_logic, Cp[i,j], Ap[i,:], Bp[:,j])
```

#### 关键特性
- **纯粹性**：用户的代码里**没有任何`T.copy`、`T.alloc_shared`等硬件相关的调用**，只关心算法本身
- **层次性**：`gemm_block_logic`可以调用`gemm_leaf_logic`，完美地表达了算法的内在层次

### 第二层：映射层 - 硬件映射文件

这一层完全模仿Cypress的`Mapping Specification`，负责告诉编译器"怎么做"。对于每一个新的硬件（M），后端专家都需要手动编写一份这样的文件。

#### 声明式映射文件
```yaml
# gemm_on_nvidia_gpu.mapping.yaml
entrypoint: gemm_host # 程序的入口任务实例

task_mappings:
  gemm_host:
    task_name: gemm_host_logic
    proc: HOST
    mems: {C: GLOBAL, A: GLOBAL, B: GLOBAL}
    tunables: {block_size: 128}
    calls: {gemm_block_logic: gemm_block_gpu} # 指定下一级调用哪个实例

  gemm_block_gpu:
    task_name: gemm_block_logic
    proc: BLOCK
    # 核心指令：告诉编译器，在这个层级，A和B必须被放到SHARED内存中
    mems: {C: REGISTER, A: SHARED, B: SHARED}
    pipeline: 3 # 启用软件流水线
```

#### 关键特性
- **解耦**：这份文件与逻辑代码完全分离，可以为同一个逻辑代码编写多个不同硬件的映射文件
- **专家驱动**：由熟悉目标硬件的专家**手动编写**，凝聚了专家的性能优化知识

### 第三层：编译器 - 智能的映射处理器

编译器是"映射指导下的代码生成"的核心，负责将"逻辑"和"映射"无缝地组合起来。

#### 编译流程详解

1. **输入**：编译器接收两个输入：`task_lang.py`（逻辑）和`gemm_on_nvidia_gpu.mapping.yaml`（映射）

2. **核心Pass：映射指导下的代码生成 (Mapping-Guided Lowering)**：
   - 编译器从入口点开始，递归遍历逻辑任务树
   - 在每个`task.launch(callee, ...)`节点：
     - **查询映射文件**：确定被调用任务应该使用哪个映射实例
     - **分析内存需求**：读取映射实例的`mems`字段，了解内存要求
     - **对比与决策**：发现内存空间不匹配时自动决策
     - **自动插入原语**：自动在生成的IR中插入必要的TileLang原语：
       - 插入`A_shared = T.alloc_shared(...)`
       - 插入`B_shared = T.alloc_shared(...)`
       - 插入`T.copy(A_from_global, A_shared)`
       - 插入`T.copy(B_from_global, B_shared)`
       - 根据`pipeline: 3`，自动将copy和计算包裹在`T.Pipelined`循环中

3. **输出**：生成一个**扁平化的、合法的、包含所有底层细节的TileLang程序**

## TVM版本的具体实现

### 阶段二：核心编译层

这是将"逻辑"和"映射"融合的地方，**其最终产物是TVM IR**。

#### 核心Pass: 映射指导下的TVM TensorIR生成

将层次化的`Task`树，**编译成一个扁平化但仍然是高层次的TVM TensorIR `PrimFunc`**。

**生成的TVM TensorIR示例：**
```python
import tvm.script as T
from tvm.tir import PrimFunc

# 这是我们的编译器"生成"的产物
@T.prim_func
def generated_kernel(
    A: T.handle, B: T.handle, C: T.handle,
    a_scratch: T.handle, b_scratch: T.handle
):
    # TIR的Buffer绑定等
    A_buffer = T.match_buffer(...)
    B_buffer = T.match_buffer(...)
    C_buffer = T.match_buffer(...)
    A_scratch_buffer = T.match_buffer(...)
    B_scratch_buffer = T.match_buffer(...)

    # 编译器根据Mapping File自动插入的DMA操作
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

这个TensorIR**和硬件特性有关**（语义上），但在**语法上是100%纯正的TVM IR**。

### 阶段三：代码生成层

#### 自定义TVM CodeGen

不必从零开始，继承TVM现有的`CodeGenC`，创建一个`CodeGenDSA`类：

```cpp
// 在我们的自定义CodeGenDSA.cc中
class CodeGenDSA : public CodeGenC {
public:
    // 重载处理外部调用的核心函数
    void VisitStmt_(const CallNode* op) override {
        // 检查外部调用的函数名
        if (op->op.same_as(StringImm("dsa.dma_copy"))) {
            // 如果是我们约定的特殊函数名
            // 就打印出对应厂商C库的函数调用字符串
            this->stream << "vendor_lib_dma_copy("
                       << this->GetVarName(op->args[0]) << ", "
                       << this->GetVarName(op->args[1]) << ");\n";
        } else if (op->op.same_as(StringImm("dsa.mma"))) {
            this->stream << "vendor_lib_mma("
                       << this->GetVarName(op->args[0]) << ", "
                       << this->GetVarName(op->args[1]) << ", "
                       << this->GetVarName(op->args[2]) << ");\n";
        } else {
            // 如果不是约定的特殊函数，调用TVM原来的默认处理方式
            CodeGenC::VisitStmt_(op);
        }
    }
};
```

#### 映射文件决定原语选择

**关键洞察**：不同厂商的C库API千差万别。因此，`Mapping File`不仅指导"参数"，更决定"原语"。

**原语库示例**：
```python
# primitives_nvidia.py
@task.leaf()
def gemm_leaf_tensorcore(C, A, B):
    # 调用NVIDIA特有的wmma指令
    T.evaluate(T.call_extern("nvidia.wmma", C, A, B, dtype="void"))

# primitives_dsa.py  
@task.leaf()
def gemm_leaf_systolic(C, A, B):
    # 调用DSA厂商提供的C库函数
    T.evaluate(T.call_extern("dsa.systolic_compute", C, A, B, dtype="void"))
```

**映射文件中的原语绑定**：
```yaml
# gemm_on_nvidia_gpu.map.yaml
task_mappings:
  gemm_block_gpu:
    task_name: gemm_block_logic
    leaf_task_bindings:
      gemm_leaf_logic: gemm_leaf_tensorcore

# gemm_on_awesomedsa.map.yaml  
task_mappings:
  gemm_block_dsa:
    task_name: gemm_block_logic
    leaf_task_bindings:
      gemm_leaf_logic: gemm_leaf_systolic
```

## 相比TileLang的优势

### 简洁性体现在前端开发模式

虽然我们的中后端逻辑与TileLang的最终目标类似（都是生成带有硬件特征的TVM TensorIR），但我们的**颠覆性优势**体现在**前端的开发模式和用户体验**上：

#### 具体场景对比：FlashAttention适配新硬件

**TileLang方式**：
- 为H100编写`flash_attention_h100.py`，硬编码性能策略
- 为AwesomeDSA适配需要**重写代码**，修改所有硬编码参数
- 结果：需要维护**两份**庞大、复杂且逻辑相似但实现细节不同的Python文件

**我们的方式**：
- 编写一次纯粹的`flash_attention_logic.py`（硬件无关）
- 为H100编写`h100.map.yaml`映射文件
- 为AwesomeDSA只需编写`awesomedsa.map.yaml`映射文件
- 算法逻辑**完全不变**，只需切换映射文件

#### 关键优势总结

1. **关注点分离**：TileLang将逻辑和策略**混合**在Python代码中，我们将其**彻底分离**
2. **工作流简化**：
   - **TileLang**：适配新硬件需要**重写并维护一份新的Python代码**
   - **我们**：适配新硬件只需要**新增并维护一份轻量级的YAML模板**
3. **可维护性**：算法逻辑升级是**一次性**的，所有硬件自动受益；性能调优是**独立**的，不影响算法逻辑

**一言以蔽之：我们让"性能模板"从需要深入代码细节才能修改的"程序"，变成了可以独立于代码、轻松配置和切换的"文件"。**

## 实现方案总结

### 完全基于TVM生态的清晰研究路径

1. **前端（我们的创新）**：构建`Task Lang` + `Mapping File`的前端，比TileLang更具层次性和解耦性
2. **中端（我们的创新）**：实现核心编译器Pass，将前端输入自动、智能地转化为**包含硬件库函数语义的、高层次的TVM TensorIR**
3. **后端（我们的创新）**：实现**轻量级的、自定义的TVM CodeGen**，将自定义的`T.call_extern`节点**翻译**成对目标DSA厂商C库的调用

### 支持M个后端的简洁性

对一个新的DSA，我们只需要：
1. 手写一份新的`Mapping File`
2. 在自定义CodeGen中，增加几个`if-else`分支，将`dsa.*`函数名映射到新厂商的C库函数名

**这个工作的复杂度，远低于从零开始为TVM编写一个完整的、过程化的CodeGen后端。**

## 未来展望：自动优化探索

### 从手动映射到智能编译

第一阶段构建了坚固的手动框架，第二阶段的目标是实现**自动化的性能调优系统**，取代专家手动编写`Mapping File`的过程。

#### 核心技术路径

1. **搜索空间参数化**：扩展`Mapping File`格式，支持可搜索的变量
   ```yaml
   tunables:
     block_size_M: !tune {type: choice, values: [64, 128, 256]}
     pipeline_depth: !tune {type: choice, values: [2, 3, 4, 5]}
   leaf_task_bindings:
     gemm_leaf_logic: !tune {type: choice, values: [gemm_leaf_tensorcore, gemm_leaf_simt]}
   ```

2. **成本模型设计**：构建基于硬件规约的分析式成本模型，结合机器学习进行精度修正

3. **搜索算法集成**：实现主调优循环，集成进化算法、模拟退火等成熟搜索算法

4. **端到端验证**：证明自动调优性能能够达到或超越专家手写的性能

通过这个渐进式的研究路径，我们将实现从手动的高性能框架到真正意义上的智能编译系统的蜕变，这是AI编译器领域的终极目标之一。
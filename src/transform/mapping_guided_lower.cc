//===----------------------------------------------------------------------===//
// SepTran Mapping-Guided Lowering Pass (Skeleton)
//
// 该文件演示如何在 TVM Pass 基础上插入自定义逻辑：
//   按 Mapping YAML 指定的信息（以 PrimFunc attr 形式传入）
//   对 TensorIR 进行 copy / pipeline 等改写。
//
// ⚠️  仅为示例骨架，尚不能独立编译运行。
//===----------------------------------------------------------------------===//

#include <tvm/ir/attr_functor.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace septran {
namespace transform {

using namespace tir;

/*! \brief 一个简单的 StmtExprMutator，后续将根据 mapping_info_ 决定插桩。*/
class MappingGuidedLowerMutator : public StmtExprMutator {
 public:
  explicit MappingGuidedLowerMutator(Map<String, ObjectRef> mapping_info)
      : mapping_info_(std::move(mapping_info)) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // TODO: 根据 attr key 决定是否插入 copy / pipeline。
    return StmtExprMutator::VisitStmt_(op);
  }

 private:
  Map<String, ObjectRef> mapping_info_;
};

/*! \brief 入口函数：接收 PrimFunc，读取其 attrs 中预先注入的 Mapping 信息。*/
PrimFunc MappingGuidedLower(PrimFunc f) {
  if (Optional<Map<String, ObjectRef>> opt = f->attrs.GetAttr<Map<String, ObjectRef>>("septran.mapping")) {
    MappingGuidedLowerMutator mutator(opt.value());
    Stmt body = mutator(f->body);
    f.CopyOnWrite()->body = body;
  }
  return f;
}

Pass MappingGuidedLowerPass() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return MappingGuidedLower(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "SepTranMappingGuidedLower", {});
}

TVM_REGISTER_GLOBAL("septran.transform.MappingGuidedLower")
    .set_body_typed(MappingGuidedLowerPass);

}  // namespace transform
}  // namespace septran
}  // namespace tvm 
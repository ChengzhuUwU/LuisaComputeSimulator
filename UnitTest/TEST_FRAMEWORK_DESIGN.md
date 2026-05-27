# IPC Physics Simulation - Unit Test Framework

## 概述

本测试框架针对基于 IPC (Incremental Potential Contact) 的物理仿真框架，聚焦于单帧牛顿迭代流程 (`NewtonSolver::physics_step_GPU` 与 `physics_step_CPU`) 的模块化单元测试。

## 设计原则

1. **解耦**: 测试模块与主仿真流程解耦，通过继承 `NewtonSolver` 访问 protected 成员
2. **可重复**: 每个测试用例独立初始化，无交叉污染
3. **可验证**: 使用有限差分 (Finite Difference) 验证解析梯度和 Hessian 的正确性
4. **分层**: 从单元测试 (能量函数) → 集成测试 (多模块协作) → 端到端测试 (完整 pipeline)

## 文件结构

```
UnitTest/
├── CMakeLists.txt                  # CMake 构建配置 (已更新)
├── xmake.lua                       # xmake 构建配置 (已更新)
├── run_tests.sh                    # 测试运行脚本
├── test_framework.h               # [新增] 测试框架: 宏、工具函数、FD 验证
├── test_base_solver.h             # [新增] TestNewtonSolverBase 基类
├── test_lbvh.cpp                  # [新增] LBVH 宽检测测试
├── test_narrow_phase.cpp          # [新增] 窄检测测试
├── test_energy_assembly.cpp       # [新增] 能量装配测试
├── test_pcg_solver.cpp            # [新增] PCG 求解器测试
├── test_ccd.cpp                    # [新增] CCD 连续碰撞检测测试
├── test_integration.cpp            # [新增] 端到端集成测试
├── test_gradient_hessian.cpp       # [已有] 能量 FD 验证 (保留, 增强)
└── test_newton_solver_integration.cpp  # [已有] 原集成测试 (保留)
```

## 测试模块详细设计

### 1. test_lbvh.cpp - LBVH 宽检测

**测试内容:**

| 测试函数 | 描述 | 验证方法 |
|---------|------|---------|
| `test_morton_ordering` | Morton code 排序正确性 | 检查树节点数、父子关系 |
| `test_aabb_reduction` | 全局 AABB 计算正确性 | 对比 LBVH root AABB vs 手工计算 |
| `test_refit` | AABB refit 正确性 | 顶点位移后验证 AABB 正确更新 |
| `test_broadphase_query_vs_gt` | 宽检测查询正确性 | 对比 LBVH 查询 vs Brute Force 结果 |
| `test_edge_tree` | Edge LBVH 构建正确性 | 验证边树节点数、类型 |
| `test_health_check` | LBVH 健康检查 | 调用 `lbvh->check_health()` |
| `test_multi_object` | 多物体 LBVH | 验证多物体场景下 LBVH 健康 |

**关键验证点:**
- Morton code 排序后相邻叶子节点在空间上相近
- AABB refit 后 root 节点的 AABB = 所有叶子 AABB 的 union
- LBVH 查询应找到 brute force 查询的超集（宽检测可能过检测）

### 2. test_narrow_phase.cpp - 窄检测

**测试内容:**

| 测试函数 | 描述 | 验证方法 |
|---------|------|---------|
| `test_vf_ccd_well_separated` | VF CCD 分离无碰撞 | 间距足够大时应无碰撞 |
| `test_vf_dcd_distance` | VF DCD 距离查询 | 调用后 narrowphase count 应为 0 |
| `test_vf_ccd_penetration` | VF CCD 穿透检测 | 穿透配置下 TOI < 1.0 |
| `test_ee_ccd_query` | EE CCD 查询 | 查询执行无异常 |
| `test_pervert_adj_list` | Per-vertex 邻接碰撞列表 | 验证碰撞列表大小合理 |
| `test_distance_vs_ground_truth` | 距离计算 vs 分析解 | 对比 `point_triangle_distance()` |
| `test_reset` | 缓冲区 reset | Reset 后计数器为 0 |

### 3. test_energy_assembly.cpp - 能量装配

**测试内容:**

| 测试函数 | 描述 | 验证方法 |
|---------|------|---------|
| `test_inertia_energy_fd` | 惯性能量 Gradient/Hessian | FD 验证 (detail 函数级) |
| `test_spring_energy_fd` | 弹簧能量 Gradient/Hessian | FD 验证, 梯度求和为零 |
| `test_stretch_face_energy_fd` | 布料 StretchFace 能量 | FD 验证, 变形后能量增加 |
| `test_neo_hookean_energy_fd` | 稳定 Neo-Hookean 能量 | FD 验证, 平移不变性 |
| `test_bending_energy_fd` | 弯曲能量 Gradient | FD 验证 (Hessian 为 GN 近似) |
| `test_abd_inertia_fd` | ABD 惯性能量 | FD 验证 |
| `test_assembly_matrix_symmetry` | 装配矩阵对称性 | 验证 Assembled cgA 对称 |
| `test_assembly_energy_consistency` | 装配能量一致性 | 静止构型下弹性能为零 |

**有限差分验证框架:**
```cpp
// 对任意能量函数 E(x) 进行 FD 验证
auto grad_fd = fd::central_difference_gradient(energy_func, x0, h=1e-4);
auto grad_ana = analytic_gradient(x0);
TEST_ASSERT_NEAR((grad_fd - grad_ana).cwiseAbs().maxCoeff(), 0.0f, 1e-3f);
```

### 4. test_pcg_solver.cpp - PCG 求解器

**测试内容:**

| 测试函数 | 描述 | 验证方法 |
|---------|------|---------|
| `test_pcg_vs_eigen_reference` | PCG vs Eigen CG | 对比求解结果残差 |
| `test_spmv_correctness` | SpMV 正确性 | 对比 CPU/Host SpMV 结果 |
| `test_solver_reset` | 缓冲区 reset | Reset 后 cgB 为零 |
| `test_real_system_convergence` | 真实系统收敛性 | 装配系统上 CG 收敛 |
| `test_eigen_solver_reference` | Eigen 直接求解器参考 | SparseLU 验证装配系统可解 |
| `test_matrix_spd_check` | 矩阵 SPD 属性 | Cholesky 分解成功验证 SPD |

### 5. test_ccd.cpp - 连续碰撞检测

**测试内容:**

| 测试函数 | 描述 | 验证方法 |
|---------|------|---------|
| `test_analytical_vf_ccd` | VF CCD 分析解 | 对比 `analytical_vf_ccd()` 函数 |
| `test_analytical_ee_ccd` | EE CCD 分析解 | 对比 `analytical_ee_ccd()` 函数 |
| `test_vf_ccd_well_separated` | CCD 分离无碰撞 | 充分间距无 TOI |
| `test_ccd_line_search_execution` | CCD 线搜索执行 | 执行后无 NaN |
| `test_ccd_fast_motion` | 快速运动防穿透 | 验证 CCD 处理快速物体 |
| `test_ccd_d_hat_margin` | d_hat 边距影响 | 增大 d_hat 后接触数增加 |
| `test_ccd_contact_energy` | 接触能量计算 | 接触能量可计算 |

### 6. test_integration.cpp - 端到端集成测试

**测试内容:**

| 测试函数 | 描述 | 验证方法 |
|---------|------|---------|
| `test_cloth_free_fall` | 布料自由落体 | 重力方向位置减小 |
| `test_cpu_gpu_consistency` | CPU/GPU 一致性 | 对比 CPU/GPU 步结果差异 |
| `test_determinism` | 仿真确定性 | 同初始状态两次运行结果相同 |
| `test_fixed_point_preservation` | 固定点保持 | 固定顶点位置不变 |
| `test_velocity_update` | 速度更新 | 重力施加后速度改变 |
| `test_predict_position` | 预测位置 | 积分后位置变化 |
| `test_multi_step_stability` | 多步稳定性 | 5 步后无 NaN |
| `test_soft_body_free_fall` | 软体自由落体 | 四面体网格在重力下下落 |

## 测试框架核心组件

### test_framework.h

提供测试基础设施:

```cpp
// 断言宏
TEST_ASSERT(condition, "message")
TEST_ASSERT_NEAR(actual, expected, tol, "message")
TEST_ASSERT_VEC3_NEAR(actual, expected, tol, "message")
TEST_ASSERT_MATRIX_NEAR(actual, expected, tol, "message")

// 计时器
ScopedTimer timer("LBVH construction");

// 有限差分验证
fd::central_difference_gradient(energy_func, x, h);
fd::validate_gradient(energy_func, x, analytic_grad, h);

// 网格生成工具
generate_cloth_grid(grid_size, vertices, faces, spacing);
generate_tetrahedron(vertices, tets);
generate_box(vertices, faces, size);
```

### test_base_solver.h

`TestNewtonSolverBase` 基类，通过继承访问 protected 成员:

```cpp
class TestNewtonSolverBase : public NewtonSolver {
    // 场景设置
    void setup_cloth_scene(int grid_size = 3, float spacing = 0.1f);
    void setup_collision_gap_scene(float gap = 0.05f);
    void setup_collision_penetration_scene();
    void setup_soft_body_scene();
    void setup_stretch_scene();

    // 数据访问器
    SimulationData<std::vector>* get_host_sim_data();
    CollisionData<std::vector>* get_host_collision_data();
    LBVH* get_lbvh_face();
    NarrowPhasesDetector* get_narrow_phase();
    ConjugateGradientSolver* get_pcg_solver();
};
```

## 构建与运行

### CMake 构建

```bash
cd build
cmake .. -DLCS_ENABLE_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target test_lbvh test_narrow_phase test_energy_assembly \
                          test_pcg_solver test_ccd test_integration -j$(nproc)
```

### xmake 构建

```bash
cd build
xmake config --lcs_enable_test=yes
xmake build -j$(nproc) test_lbvh test_narrow_phase test_energy_assembly \
                              test_pcg_solver test_ccd test_integration
```

### 运行单个测试

```bash
./build/bin/test_lbvh
./build/bin/test_energy_assembly
./build/bin/test_integration
```

### 运行所有测试

```bash
cd UnitTest
./run_tests.sh                    # 构建 + 运行所有测试
./run_tests.sh --nobuild          # 仅运行已有测试
./run_tests.sh test_lbvh          # 仅运行指定测试
```

### 运行脚本输出示例

```
╔═══════════════════════════════════════════════════════════════════════╗
║          IPC Physics Simulation - Unit Test Runner                   ║
╚═══════════════════════════════════════════════════════════════════════╝

Running all IPC framework unit tests...
------------------------------------------------------------

Running: test_lbvh
------------------------------------------------------------
  [Test] Morton code ordering...
    LBVH leaves: 9, nodes: 17
  [PASS]  test_lbvh (0.23s)

Running: test_energy_assembly
------------------------------------------------------------
  [Test] Inertia energy FD validation...
    Energy: 1.225e-05 (expected: 1.225e-05)
  [PASS]  test_energy_assembly (0.45s)

╔═══════════════════════════════════════════════════════════════════════╗
║                        Test Summary                                   ║
╚═══════════════════════════════════════════════════════════════════════╝
  test_lbvh               PASS
  test_narrow_phase       PASS
  test_energy_assembly    PASS
  test_pcg_solver          PASS
  test_ccd                PASS
  test_integration        PASS

  Total: 6 tests
  Passed: 6
  Failed: 0

╔═══════════════════════════════════════════════════════════════════════╗
║                    ALL TESTS PASSED                                   ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## 添加新测试用例

### 方式 1: 在现有测试文件中添加

```cpp
// test_energy_assembly.cpp 中添加新能量测试
bool test_new_energy_fd()
{
    std::cout << "\n  [Test] New Energy FD validation...\n";
    // ... 实现 ...
    return true;
}

// 在 main() 中注册
run(&TestEnergyAssembly::test_new_energy_fd, "New energy FD");
```

### 方式 2: 创建新的测试文件

```bash
# 1. 创建测试文件
cat > UnitTest/test_new_module.cpp << 'EOF'
#include "test_base_solver.h"
#include "test_framework.h"

class TestNewModule : public TestNewtonSolverBase {
public:
    bool test_case_1() { /* ... */ return true; }
    bool test_case_2() { /* ... */ return true; }
};

int main(int argc, char** argv) {
    // ... 标准 main 模板 ...
}
EOF

# 2. 更新 CMakeLists.txt
luisa_compute_solver_add_test(test_new_module test_new_module.cpp)

# 3. 更新 xmake.lua
target("test_new_module")
    add_rules("lc_basic_settings", {project_kind = "binary", enable_exception = true})
    add_files("UnitTest/test_new_module.cpp")
    add_deps("luisa-compute-solver-lib")
```

## 常见问题与故障排除

### 1. 编译错误: 无法访问 protected 成员

**原因**: 未继承 `TestNewtonSolverBase` 或 `NewtonSolver`

**解决**: 确保测试类声明为:
```cpp
class TestLBVH : public TestNewtonSolverBase { /* ... */ };
```

### 2. FD 验证失败: 最大误差 > 容差

**原因**:
- 能量函数包含不连续点（如铰链约束）
- Hessian 使用了 Gauss-Newton 近似
- FD 步长 h 太大或太小

**解决**:
- 调整 `fd_h` (通常 1e-4 ~ 1e-2)
- 对 GN 近似的能量只验证 Gradient
- 检查分析梯度推导

### 3. GPU 测试结果与 CPU 不同

**原因**:
- Floating point accumulation order 不同
- 原子操作竞争条件
- 同步问题

**解决**:
- CPU/GPU 差异容差设为 1e-2
- 检查 `stream() << synchronize()` 是否充分
- 验证 `device_state` 初始化正确

### 4. 测试场景数据不足

**解决**: 使用以下辅助函数生成测试数据:
```cpp
// 在 test_base_solver.h 中
generate_cloth_grid(5, vertices, faces);    // 5x5 布料
generate_tetrahedron(vertices, tets);        // 四面体
generate_box(vertices, faces);               // 立方体
```

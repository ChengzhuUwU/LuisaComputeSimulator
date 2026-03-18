多快好省的 GPU 开发

前言：跨平台 GPU 开发一二事

跨平台 GPU 计算是一件很复杂的事情。

众所周知，GPU 厂商谁也不服谁。如果你是 NVIDIA 显卡用户，可以很方便地用 CUDA，顺手再薅一把 cuBLAS、Thrust 这一套成熟生态；但一方面你得忍受 NVCC（CUDA 编译器）对构建工具链的各种限制（喜欢我 MSVC 吗），另一方面 CUDA 只能跑在 NVIDIA 显卡上。如果你想把同一套算法迁移到移动端、AMD、甚至 Apple 生态，就不得不面对 ROCm、Metal Shading Language 等一堆专有实现。

我上一份工作是做异构计算加速物理仿真的，GPU 后端是手写 Metal Shading Language。算法本身在物理上是高度通用的（理论上云端也能跑），但因为没法在别的平台上方便地运行和测试，被审稿人狠狠批了平台通用性。从那时候开始，我就一直在找：有没有一种“写法统一，又能跨平台跑”的 GPU 计算框架？

除了《无痛 CUDA 实践：MUDA Tutorial》里提到的 kokkos、taichi、LuisaCompute 之外，还有一些常见方案：
- Vulkan Compute Shader：Vulkan 本身是跨平台图形 API，我们当然可以用 Compute Shader 做计算。理想很美好，但实际写起来，GLSL 的 Shader 语法、Buffer 绑定、管线创建、Descriptor Set 管理，一整套下来代码密度极低，调试成本极高；
- OpenCL：历史悠久，但生态和维护都比较疲软了；

最后，我的选择是：用 LuisaCompute 来做物理仿真后端。

---

一、什么是 LuisaCompute？

LuisaCompute 是一个开源的、跨平台的高性能 GPU 计算框架，也是 2022 年 SIGGRAPH Asia 论文 LuisaRenderer 的底层架构。

它在我看来，是“写法最贴近现代 C++，同时还能兼顾性能和跨平台”的折中方案。诚然，在工具链完备程度上很难跟 CUDA 官方生态正面刚，但它给了 C++ 开发者一个非常干净、统一的 GPU 编程抽象，而且社区现在也非常活跃（还拿了 CCF 开源奖）。

核心思路很简单：只写 C++。LuisaCompute 允许你在纯 C++ 代码中描述 GPU 计算 Kernel，不需要引入额外的 .cu / .metal / .hlsl 文件来增加构建系统复杂度。底层做法是：在编译或运行时分析 C++ AST，把你标记的 GPU 函数 Codegen 成对应后端（CUDA、Metal、DX、Vulkan、CPU fallback）的源码或中间表示，然后再喂给各自的 Runtime。这更偏向一种 JIT（Just-In-Time Compile）+ IR 驱动的方案。

因为本质上还是 C++，你可以非常自然地用上模板、继承、泛型编程、CRTP、元编程等现代 C++ 特性，写法上会比“在字符串里写 Shader”愉快得多。

目前主要支持的后端（Backend）包括：
- cuda
- dx (DirectX 12)
- metal
- vk (Vulkan)
- fallback (CPU)

在 LuisaCompute 里写一个“矢量相加” Kernel，大致是这个味道：

```cpp
#include <luisa/luisa-compute.h>

int main(int argc, char **argv) {
    using namespace luisa::compute;

    // 初始化设备
    Context context{argv[0]};
    Device  device = context.create_device("cuda" /* or: dx, metal, vk, fallback(CPU) */);
    Stream  stream = device.create_stream(StreamTag::COMPUTE);

    uint buffer_size = 1000u;

    // 创建 Buffer
    Buffer<float> buffer_in1 = device.create_buffer<float>(buffer_size);
    Buffer<float> buffer_in2 = device.create_buffer<float>(buffer_size);
    Buffer<float> buffer_out = device.create_buffer<float>(buffer_size);

    // 上传数据
    std::vector<float> host_vector_1(buffer_size, 1.0f);
    std::vector<float> host_vector_2(buffer_size, 2.0f);
    stream << buffer_in1.copy_from(host_vector_1.data())
           << buffer_in2.copy_from(host_vector_2.data())
           << synchronize();

    // 定义 Kernel
    Shader<1> fn_add = device.compile<1>([&] {
        Var<uint>  i = dispatch_id().x;
        Var<float> x = buffer_in1.read(i);
        Var<float> y = buffer_in2.read(i);
        buffer_out.write(i, x + y);
    });

    // 启动 Kernel
    stream << fn_add().dispatch(buffer_size);

    // 下载数据
    std::vector<float> host_vector_out(buffer_size);
    stream << buffer_out.copy_to(host_vector_out.data())
           << synchronize();
}
```

整体信息密度是直接写 CUDA 的 1000000 倍，而且构建系统保持了统一的 C++ 入口。

LuisaCompute 这是一个非常有生命力的社区，现在也在非常活跃地更新与维护中，像 CUDA LLVM、适用于 Macos/Linux 的 Vulkan 后端、并行原语库 LCPP 等等的开发，都在持续演进中。

---

二、无穿透仿真器：LuisaComputeSimulator

接下来就是本篇文章的主题：我最近一直在维护的物理仿真引擎——LuisaComputeSimulator。

它是一个基于 LuisaCompute 的跨平台高性能物理模拟器（跨平台本质上全得益于 LC，我只是白嫖一下基础设施 hhhh），目前支持：
- 布料（cloth）；
- 刚体（rigid body）；
- 软体（Soft body）；
- 布料-刚体-软体耦合；
- 顶点动画与物体动画
- 基于 IPC 的无穿透接触
- Cpp 与 Python 后端等

弹性杆、MPM、流体等方向也在逐步开发中。整体求解流程采用 Newton-Raphson（牛顿迭代 / Gauss-Newton）来求解隐式时间积分下的优化问题。

---

三、从力学到优化：物理仿真是怎么变成“解优化问题”的？

关于变形体仿真的背景，强烈推荐几个课程：Games 103、Dynamic Deformables、Physics-Based Simulation 等，这里只抓住最核心的几个式子。

### 3.1 自由度与状态变量

对于 Deformable 物体（会产生顶点位移动画的物体，比如布料、软体、弹性杆等），通常把每个顶点的位置作为自由度：
- 顶点个数为 $n_v$，
- 每个顶点有 3 个方向的位移（x, y, z），
- 系统自由度 $n = 3 n_v$。

对于刚体，如果用经典的 6 自由度表述（3 平移 + 3 旋转参数），则：
- 刚体个数为 $n_b$，
- 系统自由度 $n = 6 n_b$。

在 LuisaComputeSimulator 里，刚体部分我们后来会改成 Affine Body Dynamics（ABD）的 12 自由度形式，这个后面再详细展开。

### 3.2 隐式欧拉时间积分 → 能量最小化

考虑从时间 $t^n$ 到 $t^{n+1} = t^n + \Delta t$ 的一步积分。经典的隐式欧拉写法是：

$$
\begin{aligned}
\mathbf{v}^{n+1} &= \mathbf{v}^n + \Delta t\,\mathbf{M}^{-1}\mathbf{f}(\mathbf{x}^{n+1}), \\
\mathbf{x}^{n+1} &= \mathbf{x}^n + \Delta t\,\mathbf{v}^{n+1},
\end{aligned}
$$

其中 $\mathbf{x}$ 是所有自由度拼起来的位置向量，$\mathbf{v}$ 是速度，$\mathbf{M}$ 是质量矩阵，$\mathbf{f}$ 是合力（通常由各种势能的梯度给出）。

把 $\mathbf{v}^{n+1}$ 消掉，可以把隐式欧拉写成对 $\mathbf{x}^{n+1}$ 的一个能量最小化问题：

$$
\min_{\mathbf{x}^{n+1}} \
\Phi(\mathbf{x}^{n+1})
= \frac{1}{2 \Delta t^2}(\mathbf{x}^{n+1} - \tilde{\mathbf{x}})^\mathrm{T} \mathbf{M} (\mathbf{x}^{n+1} - \tilde{\mathbf{x}})
+ E_{\text{potential}}(\mathbf{x}^{n+1}).
$$

这里：
- $\tilde{\mathbf{x}} = \mathbf{x}^n + \Delta t\,\mathbf{v}^n$ 是“显式外推”的位置；
- 第一项可以看作是一种“动能”或“惯性势能”，结构上和大家熟悉的 $\tfrac{1}{2}m v^2$ 很像；
- $E_{\text{potential}}$ 汇总了所有势能项：拉伸、弯曲、重力位能、碰撞能量等等。

“隐式”的好处是：
- 在较大时间步长（例如 $\Delta t \ge 1/100\,\text{s}$）下仍然数值稳定，不容易爆炸；
- 代价是每一步都要解一个非线性优化问题，而不是像显式欧拉那样 O(能量项数) 地直接更新一次位置。

### 3.3 Newton-Raphson / Gauss-Newton 迭代

给定能量 $\Phi(\mathbf{x})$，Newton-Raphson 迭代的形式是：

$$
\mathbf{x}^{k+1} = \mathbf{x}^{k} - \mathbf{H}(\mathbf{x}^k)^{-1} \, \nabla \Phi(\mathbf{x}^k),
$$

其中：
- $\nabla \Phi$ 是梯度（gradient），
- $\mathbf{H} = \nabla^2\Phi$ 是 Hessian 矩阵（$n \times n$ 的稀疏对称矩阵）。

实际实现里，我们通常不显式求逆，而是每次解线性方程：

$$
\mathbf{H}(\mathbf{x}^k) \Delta \mathbf{x}^k = -\nabla \Phi(\mathbf{x}^k),
$$

然后更新 $\mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k$。

在 LuisaComputeSimulator 中：
- 我们把所有能量项统一写成 $E_i(\mathbf{x})$，
- 遍历每个能量，计算对应的局部 Gradient 和 Hessian block，
- 通过装配过程累加到全局 $\mathbf{g} = \nabla\Phi$ 和 $\mathbf{H}$ 上。

除了牛顿法，也有很多其他非线性优化方案，比如：
- 纯梯度下降（Gradient Descent），2016 年有论文用它来做物理仿真；
- 各种块坐标下降（如 VBD、SOSD）；
- LBFGS 等拟牛顿法；
- ADMM 一类的分布式/约束优化方法。

比如梯度下降：

$$
\mathbf{x}^{k+1} = \mathbf{x}^k - \alpha \, \nabla \Phi(\mathbf{x}^k),
$$

这里 $\alpha$ 就是机器学习里熟悉的 learning rate。假设有个顶点连了 6 根弹簧，把 $\alpha$ 固定取成 $1/6$ 的那种做法，可以类比成线性方程组里的 Jacobi 迭代：

$$
\mathbf{x}^{k+1} \approx \mathbf{x}^k + \frac{1}{\sum_j A_{ij}} (\mathbf{b} - A\mathbf{x}^k).
$$

这些方法实现起来更轻量，单次迭代只需要向量级别的运算，但一般需要更小步长和更多迭代才能收敛。LuisaComputeSimulator 目前主力还是牛顿/PCG 这一套经典 pipeline。

---

四、线性方程组与 PCG：核心数值引擎

牛顿法的每一步，都要解一个巨大的稀疏线性方程组：

$$
\mathbf{H} \Delta \mathbf{x} = -\mathbf{g},
$$

$\mathbf{H}$ 的维度等于系统自由度（几十万甚至上百万），直接用 Cholesky 之类的直接法几乎不现实，所以我们用迭代法。LuisaComputeSimulator 里使用的是预处理共轭梯度法（Preconditioned Conjugate Gradient，PCG），它属于 Krylov 子空间方法的一类。

简单回顾一下 PCG 的核心：
- 初始给一个方向 $\mathbf{p}_0$，残差 $\mathbf{r}_0 = \mathbf{b} - \mathbf{A}\mathbf{x}_0$；
- 每次迭代在 Krylov 子空间里找一个新的搜索方向 $\mathbf{p}_{k+1}$，
- $\mathbf{p}_k$ 和 $\mathbf{p}_{k+1}$ 之间保持 $\mathbf{A}$-正交；
- 通过预处理矩阵 $\mathbf{M}^{-1}$ 把问题变成 $\mathbf{M}^{-1}\mathbf{A}$ 更好解的形式。

理论上，如果预处理矩阵 $\mathbf{M}$ 等于 $\mathbf{A}^{-1}$，那只要一步就能收敛。但显然这相当于先把问题解完再求解一遍……所以实际预处理只能是一个“近似”，比如：
- 对角预处理（Jacobi），
- 分块对角预处理，
- 或更高级的 AMG/ILU 等。

在实际的仿真场景里：
- 线性系统自由度非常高（例如 10 万顶点的布料，自由度 30 万），
- 每个牛顿迭代里，PCG 常常需要几十到几百步迭代，
- PCG 本身往往是整条 pipeline 里最耗时的部分。

### 4.1 GPU 上的 SPMV：ReduceByKey 风格的实现

在 GPU 上实现 PCG，最核心的算子有两个：
- 稀疏矩阵-向量乘法（SPMV，Sparse Matrix-Vector Multiply），
- 各种归约（内积、范数计算等）。

LuisaComputeSimulator 目前对全局稀疏矩阵没有显式构建成 CSR/COO 格式的“一个大矩阵”，而是：
- 把每个能量项产生的 Hessian 局部块写成若干 triplet（行、列、值），
- 再通过排序 + 累加，把同一对自由度上的块叠加起来，逻辑上等价于构建了一个全局矩阵，
- SPMV 则在这些 triplet 上做“按 key 归约”的运算。

一个典型的 GPU SPMV 流程可以抽象成 ReduceByKey：
1. 先构造一个数组，元素形如 $(i, j, A_{ij})$；
2. 对这个数组按行号（有时也会连同行列一起 encode 成一个 key）做排序；
3. 每个线程处理一个或一小段非零项，临时计算 $A_{ij} x_j$；
4. 在同一行内用 Block-Level 归约（reduce by key）把所有 $A_{ij} x_j$ 累加成 $y_i$；
5. 写回结果向量 $\mathbf{y} = \mathbf{A}\mathbf{x}$。

这样做的好处是：
- 不必为每一种能量专门定制一个大矩阵装配格式，
- 很适合 GPU 的并行粒度（一个 non-zero 一条线程），
- 可以在局部 Block 内用共享内存完成很多合并，减少原子操作。

当然，这个实现现在还算是“工程可用但不是极致最优”的形态，后面会提到碰撞 Hessian 装配里的两次 Block-Level 排序技巧，以及未来打算迁到 LCPP 上做更彻底的优化。

---

五、本构模型：从布料到 Affine Body Dynamics

LuisaComputeSimulator 目前在布料、软体、刚体方面，使用了几类不同的本构（constitutive）模型。

### 5.1 布料本构

布料部分主要有两条线：
- 质点-弹簧模型（mass-spring）：可以参考 2002 年的 Stable but Responsive Cloth 一类工作；
- 线弹性有限元模型：参考的是 2019 年的 finite element formulation of bar / shell 相关工作（这里非常感谢原作者开源代码，基本可以直接阅读和移植）。

质点-弹簧模型实现和理解都比较直观，适合 demo 和入门；
线弹性 FEM 则更加“物理一点”，尤其在要统一和软体仿真时更自然。

### 5.2 软体本构：Stable Neo-Hookean

软体我们使用的是 stable Neo-Hookean 材料，这是一类超弹性有限元材料模型。它的特点是：
- 能量写在形变梯度 $\mathbf{F}$ 或 Cauchy-Green 张量上，
- 对大形变有较好的数值稳定性，
- 但也会给牛顿迭代带来比较强的非线性（Hessian 更“硬”）。

在实现上，依然是把每个四面体（或体元）的局部能量 $E(\mathbf{F})$ 展开成对顶点自由度的梯度和 Hessian block，然后装配到全局系统中。

### 5.3 刚体本构：Affine Body Dynamics（ABD）

刚体部分，我们选用的是 2022 年的 Affine Body Dynamics（ABD） 思路。直观地说：
- 传统刚体通常用 6 个自由度（3 平移 + 3 旋转参数）来描述，
- 而 ABD 用一个 3×3 的线性变换矩阵 $\mathbf{A}$ 加上一个平移 $\mathbf{t}$，总共 12 个自由度来描述刚体：

$$
\mathbf{x}(\mathbf{X}) = \mathbf{A} \, \mathbf{X} + \mathbf{t},
$$

其中 $\mathbf{X}$ 是刚体在参考构型下的顶点位置，$\mathbf{x}$ 是当前世界空间位置。

这样做有几个好处：
1. **消除旋转参数化的奇异性**：不再纠结欧拉角万向节锁、四元数归一化等问题，$\mathbf{A}$ 只是一个线性算子；
2. **方便做线性的 CCD**：顶点轨迹在一步时间里是仿射的，$\mathbf{x}^{n+1}(t)$ 随时间 $t$ 线性或分段线性变化，这对连续碰撞检测（CCD）非常友好，可以在参数空间上做线性/多项式求根；
3. **方便和软体管线做耦合**：刚体的能量形式长得跟一个“刚度很大的软体”非常像，只不过它的自由度不再是每个顶点，而是刚体的 12D 仿射参数；
4. **允许轻微的弹性形变**：如果刚度设置得非常大，理想刚体会保持 $\mathbf{A}$ 接近正交矩阵；但 ABD 允许一定的拉伸 / 压缩，这在真实材料里反而更合理，也有助于和有限元软体做统一处理。

在 LuisaComputeSimulator 里：
- 每个刚体都有自己的 affine 自由度，
- 网格顶点通过一个线性的 Jacobian 把力和能量映回到刚体自由度上，
- 后续 joint 约束等也可以写成能量，再通过同一套 Jacobian 传播。

---

六、碰撞：DCD / CCD、能量形式与 Hessian 装配

碰撞检测和响应是整个管线中最复杂、也是最能拉开 GPU / CPU 差距的一环。

从宏观上看，我们的碰撞模块分为：
- DCD（Discrete Collision Detection）：每一帧或每个牛顿迭代只看离散的位姿；
- CCD（Continuous Collision Detection）：考虑一个时间步内部物体是“连续动起来”的情况，用于避免隧穿；
- 碰撞能量与梯度 / Hessian 的装配（IPC 风格的 Log-Barrier 能量）。

### 6.1 DCD：VF / EE 检测与退化情形

在 DCD 阶段（也就是“几何检测”），我们主要做两类基本原语的距离检测：
- VF：顶点-三角形（Vertex-Face），
- EE：边-边（Edge-Edge）。

这两类已经能覆盖大多数三维网格之间的接触情况。实际实现时会遇到退化情形：
- VF 退化成 VV（点-点），
- EE 退化成 VE（点-边）。

在 IPC 体系中，通常会把它们区分开来；但在 LuisaComputeSimulator 中，我们采用了一个更统一的做法：
- 始终以 VF / EE 的形式存储一条“碰撞约束”，
- 如果几何上退化成了 VV/VE，就把对应顶点的重心坐标直接设为 0，
- 在一次牛顿迭代中，我们把这些重心坐标视为常量（Gauss-Newton 风格的近似），
- 这样距离对各个顶点自由度的偏导只依赖于：重心坐标和法向（碰撞方向），而不显式区分“这到底是 VF 还是 VV”。

距离检测和 CCD 的写法主要参考了 ZOZO 的 ppf-contact-solver 项目，CCD 部分则参考了 2021 年 C-IPC（Conditional IPC）中的 Adaptive CCD 方案。

### 6.2 CCD：C-IPC 风格的流程

如果只做 DCD，很容易出现“隧穿”：
- 物体在单步内移动距离大于安全距离 $\hat d$，
- 离散时刻看起来“在障碍物两侧”，
- 中间那一瞬间穿过去了但没被检测到。

为了解决这个问题，IPC 系列工作一般会配套 CCD：
- 给定上一时刻和当前预测位姿，
- 在时间参数 $t \in [0, 1]$ 上做连续的 VF / EE 检测，
- 找到最小的 TOI（time of impact），
- 把这条约束加入优化问题中，限制物体不能越过这个 TOI。

严格的 Global CCD 是：
- 在全局所有候选对里找一个最小 TOI，
- 以这个 TOI 为准回退整体步长。

这样物理上最干净，但数值上很拖慢收敛。实践中很多系统会在 Global CCD 和 Local CCD（逐顶点 TOI）之间做权衡。LuisaComputeSimulator 当前采用的 CCD 流程大体上跟 C-IPC 的 Adaptive CCD 类似：
- 在几何上做一定的分层 / 分批处理，
- 在保证收敛性的前提下尽量减少“全局最小 TOI”带来的过度保守。

### 6.3 碰撞能量的几种形式：二次、三次、Log、Log-Barrier

检测到一对潜在碰撞对之后，怎么把“不能相互穿透”写成一个能量？

设 $d$ 是几何距离，$\hat d$ 是安全距离（通常在 1–20 mm 之间），定义约束：

$$
C = d - \hat d.
$$

我们希望在 $C < 0$（太近 / 穿透）的时候惩罚它，在 $C \ge 0$ 的时候能量和力都为 0。常见的一些能量形式：

1. **二次型能量（Quadratic）**

$$
E_{\text{quad}} = \frac{1}{2} \kappa \, \max(0, -C)^2,
$$

其中 $\kappa$ 是劲度系数，通常在 $10^6 \sim 10^{10}$ 的量级。

优点：
- 简单，导数也非常好算；

缺点：
- 提供的最大斥力有限，数量级大约是 $\kappa \hat d$，
- 在有巨大外力或质量特别大的物体时，可能给不出足够的反作用力，
- 即便配合 CCD 做回退，仍可能出现卡死或轻微穿透。

2. **三次型 / 更高次多项式能量**

有些工作会把能量改成三次或更高次，以在靠近 $d = 0$ 的时候提高增长速度，但本质问题依然存在：
- 只要是多项式，能量和力的增长都是“有限”的，
- 真正挤到一起时，还是不如“趋近无限大”的势垒函数。

3. **简单 Log 形式：$-\log(d / \hat d)$**

另一种思路是直接用对数势能：

$$
E_{\log} = -\kappa \, \log\left(\frac{d}{\hat d}\right),
$$

或者变体形式。对数在 $d \to 0$ 时会趋向 $+\infty$，
理论上可以提供“无上限”的斥力。但如果直接用对数，会在 $d = \hat d$ 处产生不够光滑的问题。

4. **Log-Barrier（IPC 使用的平滑势垒函数）**

IPC 采用的是一种平滑后的 Log-Barrier 形式：

$$
E_{\text{IPC}} = -\kappa (d - \hat d)^2 \log\left(\frac{d}{\hat d}\right).
$$

这个形式有几个非常关键的好处：
- 当 $d \to 0$ 时，对数项让能量趋向无穷大，可以提供“无限大”的理论斥力；
- $d \to \hat d$ 时，前面的 $(d - \hat d)^2$ 会把能量、梯度、Hessian 都平滑到 0；
- 整个能量关于距离是 $C^2$ 连续的，非常适合牛顿迭代。

在 LuisaComputeSimulator 里，我们主要采用的就是这种 Log-Barrier 能量。

需要强调的是：“无限大”在实现中并不是真的无限：
- 当 $d < 10^{-5}\,\text{m}$（比如 0.01mm 量级）时，浮点精度已经很难可靠地区分更小的距离；
- 即使能量公式写着 $+\infty$，实际算出来也会因为 underflow / overflow 或精度误差而失真；
- 所以实践中往往需要配合调整 $\kappa$、启用 CCD、适当做步长控制等手段，不能指望单纯靠 Log-Barrier“自动解决一切”。

### 6.4 碰撞 Hessian 装配：两次 Block-Level 排序的小技巧

碰撞能量的 Hessian 是装配里最重的一块：
- 每个 VF / EE 约束都贡献一个 4×4 或 8×8 的小 block，
- 非对角 block 数量巨大，
- 直接用全局原子加（atomicAdd）会把性能打到地板上。

LuisaComputeSimulator 目前采用了一种“折中工程方案”：
- 先按 block 所在的“行块索引”做一次 Block-Level 排序，
- 对同一个 Block-Row 内的 block，用共享内存做一次局部归并，把相同 (i, j) 的非对角块加在一起；
- 再按“列块索引”做第二次 Block-Level 排序，在另一个维度上进一步合并；
- 合并后的结果写入一个中间缓冲或最终的 Hessian 结构。

这种方案的优点：
- 不需要写完整的全局 RadixSort + 归并逻辑，工程量相对可控；
- 不少重复 block 可以在 Block 内被消掉，减少原子操作和内存带宽压力；

但它也不是理论最优：
- 仍然会有相当一部分“跨 Block 的重复非对角块”没能被合并；
- 总体上牺牲了一些峰值性能，换取实现难度的降低。

后续的计划是：
- 把这部分重写到另一个项目 LCPP 里，
- 利用更优化的稀疏矩阵和排序 / 归并基础设施，
- 做一次“真正意义上的全局装配加速”，敬请期待。

---

七、LuisaCompute 部分：并行原语、Dynamic Resize 与多线程 JIT

前面从物理和数值层面讲了很多，这一节回到“怎么在 LuisaCompute 里把这些东西跑起来”。

### 7.1 并行原语：Reduce / Scan / Sort

很多地方我们都需要用到基础并行原语：
- 向量点乘（PCG 里的 inner product），
- 前缀和（Prefix-Sum / Scan），例如构造碰撞列表时的压缩写入，
- 排序（Sort），例如碰撞约束按 key 排序、Hessian block 合并等。

目前我在项目里主要实现了一些 Block-Level（线程块级别）的 Reduce / Scan / Sort：
- 优点是实现相对简单，也比较符合 LuisaCompute Kernel 的风格；
- 缺点是在需要全局 Reduce / Scan / Sort 的地方会显得吃力，比如需要多轮 Kernel 启动来收集跨 Block 的信息。

好消息是：Ligo 老师最近在开发一套基于 LuisaCompute 的通用并行原语库，其中 Scan / Sort 使用了 Decoupled Lookback 这种经典方案：
- 避免了频繁启动小 Kernel 去拿“前面 Block 的中间结果”，
- 通过全局内存里的一些 metadata 来实现跨 Block 的前缀和；
- 但也因此需要用到比较底层的指令（volatile / thread_fence 等），多后端调试成本不小——目前也还在持续打磨中，值得大家关注。

### 7.2 Dynamic Resize：动态缓冲与 BindingGroup

碰撞相关的数据结构有一个共同特点：
- 数量是动态变化的，
- 很难一开始就预估一个“小而刚好”的上界。

典型的例子包括：
- BroadPhase（宽相）之后的候选碰撞对（通过 BVH AABB 剔除得到），
- NarrowPhase（窄相）几何检测之后的 VF / EE 对，
- 如果扩展到沙子 / 弹性杆，还会有 VV / VE 等对，
- 装配后的 Hessian triplet / block 列表。

直觉上我们当然可以“一刀切”地给每个 Buffer 乘以一个大常数，分配一个巨大的空间，但这在 GPU 显存和 cache 层面都非常浪费。所以实际做法是：
- 碰撞检测时记录当前写入位置，
- 如果发现超过了 Buffer 当前容量，就做一次动态 Resize；
- 下一轮迭代在更大的 Buffer 上继续用。

但这会带来一个问题：
- 如果 Kernel 里是以“捕获的 BufferView”方式访问内存，
- Resize 之后，原来被捕获的指针就失效了；
- 这在 JIT / 多线程编译场景下非常容易踩坑。

一个可行方案是：
- 把这类动态 Buffer 改用“参数传递”的方式（更像 CUDA 的裸指针传参），
- Kernel 每次启动前都传入最新的 Buffer 句柄；
- 代价是写法稍微啰嗦一点。

另一个方案是：
- 使用 LuisaCompute 的 BindingGroup 把相关 Buffer 组织在一个结构体里；
- 本质上还是逐个绑定，但用户侧访问更模块化；
- 当某个 Buffer Resize 之后，只要在启动前重新绑定一次 BindingGroup；
- 非常适合做能量、材料、碰撞等子系统的解耦。

需要注意的是：
- 因为 BindingGroup 会一次性绑定其中的多个 Buffer，
- 在某些后端（例如 DX）上会受到“绑定参数总大小”的限制（例如总指针大小不能超过 64 Byte），
- 所以在设计 BindingGroup 结构时，也要兼顾这些硬件约束。

目前 LuisaComputeSimulator 里，像惯性势能、拉伸能量、弯曲能量等模块，已经逐步迁移到 BindingGroup 形式，后续也会继续朝这个方向整理。

### 7.3 多线程 JIT：80+ Kernels 的冷启动问题

物理仿真涉及非常多的 kernel：
- 各种能量项的 Gradient / Hessian 计算，
- 碰撞检测（BroadPhase / NarrowPhase / CCD），
- 各种 PCG/线性代数算子，
- 以及一堆辅助工具 Kernel。

项目里 Kernel 总数在 80 个以上，如果全部在第一次运行时串行 JIT 编译，冷启动体验非常糟糕，尤其在 CUDA 后端：
- NVCC 本身就比较慢，
- 再叠加 JIT 和多后端支持，等几秒钟甚至十几秒都不稀奇。

为了解决这个问题，麦老师帮忙写了一套多线程 JIT 流程：
- 在程序启动时，把所有 Kernel 的编译任务分发到多个线程；
- 对于 DX / Metal 这类后端，整体 JIT 时间可以压到 1s 左右，几乎是“秒开”；
- CUDA 后端因为 NVCC 太拉了还是会慢一些，但也已经好很多。

这里有一个小坑：
- 在多线程环境下，如果用“引用捕获”的 Lambda 去描述 Kernel，
- 很容易因为 Lambda 捕获的上下文已经失效 / 被移动，而出现莫名其妙的问题；
- 更安全的做法是统一改成值捕获。

长期目标的话，我个人还是希望能做到“比较纯的 AOT” 方案：
- 尽量把 Kernel 内的 Buffer 访问从捕获 BufferView 改成绑定 / 参数形式；
- 通过离线 CodeGen + 多后端工具链，把大部分代价前移到编译阶段。

---

结语

写到这里差不多就是这一年折腾 LCS 的一个小结。

最近两年没有在知乎上写东西了，似乎一直在忙论文。。。（虽然忙了这么久只有一篇），而我的工作效率也不算高，经常需要对着电脑发呆半天，才能把一个公式或者一个能量项想清楚。

但回过头看，这套“多快好省的 GPU 开发 + 物理仿真管线”还是带来了不少成就感：
- 在 C++ 里统一表达 GPU / CPU 计算，
- 在 LuisaCompute 上把 IPC / ABD / Neo-Hookean 这些比较前沿的仿真技术串起来，
- 在工程上把动态碰撞、Log-Barrier 能量、SPMV / PCG、Hessian 装配等细节一一落地。

后面还会继续填坑（比如更好的预条件子、更高阶的能量模型、LCPP 加速的装配与线性代数库、关节约束、弹性杆、体网格等等），也欢迎感兴趣的同学一起交流、提 issue、顺手帮忙踩坑。

如果你也有“写一套自己的物理引擎 / GPU 计算框架”的想法，希望这篇文章能给你一点点启发，或者至少帮你少踩几个我已经踩过的坑。

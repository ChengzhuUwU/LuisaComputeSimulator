多快好省的 GPU 开发

前言：跨平台 GPU 计算是一件很复杂的事情

众所周知，GPU 的架构差别很大。如果你是 NVIDIA 显卡用户，那么你可以很方便地使用 CUDA 并享受 CUBlas 这样的强大工具链，当然，一方面是你需要忍受 NVCC （CUDA 的编译器）对于构建工具链的限制，另一方面，CUDA 只能运行在 NVIDIA 的显卡上面，如果你想将其迁移到移动端、AMD 的显卡、或者苹果设备上，则需要转化为专有的实现，比如 AMD 上的 ROCm，MacOS/IOS 上的 Metal Shading Language 等等。

我上一篇工作是做异构计算加速物理仿真的，GPU 后端是手写 Metal Shading Language。算法本身在物理上是通用的（理论上云端也能跑），但因为不能在其他的设备上运行与测试，被审稿人狠狠批了平台通用性。所以我也在思考是否有一个跨平台的 GPU 计算框架来实现这个事情。

文章 [无痛CUDA实践：MUDA Tutorial](https://zhuanlan.zhihu.com/p/659664377) 列举了一些常用的跨平台计算方案，比如 kokkos、taichi、LuisaCompute 等，此外还包括下面的方案，这些方案都各有优劣：

- Vulkan Compute Shader：Vulkan 本身是跨平台图形 API，我们当然可以用 Compute Shader 做计算。理想很美好，但实际写起来，GLSL 的 Shader 语法、Buffer 绑定、管线创建、Descriptor Set 管理，一整套下来代码密度极低，调试成本极高；
- OpenCL：历史悠久，但生态和维护都比较疲软了；

最后我的选择是用 LuisaCompute 来做物理仿真。

---

一、什么是 LuisaCompute？

LuisaCompute 是一个开源的、跨平台的高性能 GPU 计算框架，也是 2022 年 SIGGRAPH Asia 论文 LuisaRenderer 的底层架构。项目也获得了 [2025 年 CCF-CAD&CG 优秀图形开源软件](https://www.ccf.org.cn/Chapters/TC/TC_Listing/TCCADCG/hyxw/2025-08-27/848089.shtml)。

LuisaCompute 是我认为写法上最简洁，就同时兼顾着性能与通用性的方案，给了 C++ 开发者一个非常干净、统一的 GPU 编程抽象。虽然论文的早起开发者主要来自渲染领域，但是这不影响将这个框架迁移到物理仿真中。

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

    // 初始化设备
    luisa::compute::Context context{argv[0]};
    luisa::compute::Device  device = context.create_device("cuda" /* or: dx, metal, vk, fallback(CPU) */);
    luisa::compute::Stream  stream = device.create_stream(StreamTag::COMPUTE);

    uint buffer_size = 1000u;

    // 创建 Buffer
    luisa::compute::Buffer<float> buffer_in1 = device.create_buffer<float>(buffer_size);
    luisa::compute::Buffer<float> buffer_in2 = device.create_buffer<float>(buffer_size);
    luisa::compute::Buffer<float> buffer_out = device.create_buffer<float>(buffer_size);

    // 上传数据
    std::vector<float> host_vector_1(buffer_size, 1.0f);
    std::vector<float> host_vector_2(buffer_size, 2.0f);
    stream << buffer_in1.copy_from(host_vector_1.data())
           << buffer_in2.copy_from(host_vector_2.data())
           << synchronize();

    // 定义 Kernel
    luisa::compute::Shader<1> fn_add = device.compile<1>([&] {
        luisa::compute::Var<uint>  i = luisa::compute::dispatch_id().x;
        luisa::compute::Var<float> x = buffer_in1.read(i);
        luisa::compute::Var<float> y = buffer_in2.read(i);
        buffer_out.write(i, x + y);
    });

    // 启动 Kernel
    stream << fn_add().dispatch(buffer_size);

    // 下载数据
    std::vector<float> host_vector_out(buffer_size);
    stream << buffer_out.copy_to(host_vector_out.data())
           << luisa::compute::synchronize();
}
```

非常简洁的实现，信息密度是 CUDA 的 114514 倍，而且构建系统保持了统一的 C++ 入口。

诚然，虽然当前的生态链无法与强大的 CUDA 相比，但这是一个非常有生命力且开放的社区，现在也在非常活跃地更新与维护中，像 CUDA LLVM、适用于 Macos/Linux 的 Vulkan 后端、并行原语库 LCPP 等等的开发，都在持续演进中。

---

二、无穿透仿真器：LuisaComputeSimulator

因此我也非常激动地向大家介绍最近半年以来在维护的物理仿真引擎——LuisaComputeSimulator。

它是一个基于 LuisaCompute 的跨平台高性能物理模拟器（跨平台本质上全得益于 LC，我只是白嫖一下基础设施 hhhh），目前支持：
- 布料（cloth）；
- 刚体（rigid body）；
- 软体（Soft body）；
- 布料-刚体-软体耦合碰撞；
- 顶点动画与物体动画
- 基于 IPC 的无穿透碰撞响应
- 高精度摩擦建模
- Cpp 与 Python 后端等

弹性杆、MPM、流体、Joint Containt 等功能也在逐步开发中。

整体求解流程采用 Newton-Raphson 迭代（简称牛顿迭代）来求解隐式时间积分下的优化问题，使用 SIGGRAPH 2020 年论文 [Incremental Potential Contact（简称 IPC）](https://zhuanlan.zhihu.com/p/154542103) 来处理碰撞与磨擦接触。

---

三、什么是物理仿真中的优化问题

感兴趣的读者可以几个课程：
- [Games 103](https://www.bilibili.com/video/BV12Q4y1S73g/)
- [Dynamic Deformables](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/)
- [Physics-Based Simulation](https://phys-sim-book.github.io/)

这里仅作简要介绍：

### 3.1 自由度与状态变量

对于 Elastic 物体（会产生顶点位移动画的物体，比如布料、软体、弹性杆等），通常把每个顶点的位置作为自由度：
- 顶点个数为 $n_v$，
- 每个顶点有 3 个方向的位移（x, y, z），
- 系统自由度 $n = 3 n_v$。

对于刚体，如果用经典的 6 自由度表述（3 平移 + 3 旋转参数），则：
- 刚体个数为 $n_b$，
- 系统自由度 $n = 6 n_b$。

不过考虑到 6 自由度的表示存在万向锁与非线性运动路径的问题，因此我们使用的是 12 自由度形式，方法来自 SIGGRAPH 2023 论文 [Affine Body Dynamics（ABD）](https://dl.acm.org/doi/10.1145/3528223.3530064)。

### 3.2 隐式欧拉时间积分 与 能量最小化

考虑从时间 $t^n$ 到 $t^{n+1} = t^n + \Delta t$ 的一步积分。经典的隐式欧拉写法是：

$$
\begin{aligned}
\mathbf{v}^{n+1} &= \mathbf{v}^n + \Delta t\,\mathbf{M}^{-1}\mathbf{f}(\mathbf{x}^{n+1}), \\
\mathbf{x}^{n+1} &= \mathbf{x}^n + \Delta t\,\mathbf{v}^{n+1},
\end{aligned}
$$

其中 $\mathbf{x}$ 是所有自由度组成的状态向量，$\mathbf{v}$ 是速度，$\mathbf{M}$ 是质量矩阵（其中 Elastic Body 部分是对角矩阵，Rigid Body 部分 是 12x12 的对角块矩阵），$\mathbf{f}$ 是每个自由度上的合力（包括外力与保守的内力，通常由各种势能的梯度给出）。

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
- $E_{\text{potential}}$ 汇总了所有势能项：拉伸、弯曲、重力势能、碰撞能量、摩擦能量等等。

隐式积分的特点是：
- 在较大时间步长（例如 $\Delta t \ge 1/100\,\text{s}$）下仍然数值稳定，不容易爆炸；
- 代价是每一步都要解一个非线性优化问题，而不是像显式欧拉那样 O(能量项数) 地直接更新一次位置。不过考虑到基于显式积分的模拟方法的步长通常要选取的很小，所以实际上隐士积分更加高效。Verlet 积分就是一个著名的显式积分方法。

### 3.3 Newton-Raphson / Gauss-Newton 迭代

给定优化目标 $F(\mathbf{x})$，Newton-Raphson 迭代的形式是：

$$
\mathbf{x}^{k+1} = \mathbf{x}^{k} - \mathbf{H}(\mathbf{x}^k)^{-1} \, \nabla F(\mathbf{x}^k),
$$

其中：
- $\nabla F$ 是梯度（gradient），
- $\mathbf{H} = \nabla^2 F$ 是 Hessian 矩阵（$n \times n$ 的稀疏对称矩阵）。

在实现中，我们通常不显式求逆（因为很容易受到浮点精度的影响，且求逆后的矩阵是稠密矩阵，我们的内存/显存可能完全存储不下），而是每次解线性方程：

$$
\mathbf{H}(\mathbf{x}^k) \Delta \mathbf{x}^k = -\nabla F(\mathbf{x}^k),
$$

然后更新 $\mathbf{x}^{k+1} = \mathbf{x}^k + \Delta \mathbf{x}^k$。

在 LuisaComputeSimulator （LCS） 中：
- 我们把所有能量项统一写成 $E_i(\mathbf{x})$，
- 遍历每个能量，计算对应的局部 Gradient 和 Hessian block，
- 通过装配过程累加到 全局梯度向量 $\nabla F$ 与全局 Hessian $\nabla^2 F$ 上。

除了牛顿法，也有很多其他非线性优化方案，比如：
- 梯度下降法（Gradient Descent）
- 块坐标下降（如 VBD、SOSD）
- LBFGS 等拟牛顿法
- ADMM 一类的分布式/约束优化方法

比如梯度下降：

$$
\mathbf{x}^{k+1} = \mathbf{x}^k - \alpha \, \nabla F(\mathbf{x}^k),
$$

这里 $\alpha$ 就是机器学习里熟悉的 learning rate。假设有个顶点连了 6 根弹簧，把 $\alpha$ 固定取成 $1/6$ 的那种做法，可以类比成线性方程组里的 Jacobi 迭代：

$$
\mathbf{x}^{k+1} \approx \mathbf{x}^k + \frac{1}{\sum_j A_{ij}} (\mathbf{b} - A\mathbf{x}^k).
$$

这些方法实现起来更轻量，单次迭代只需要向量级别的运算，有一篇 2016 年的经典论文 Descent Methods for Elastic Body Simulation on the GPU 正是用梯度下降法做物理仿真，在 GPU 上有更好的并行性。但一般需要更小步长和更多迭代才能收敛。LuisaComputeSimulator 选用的是 牛顿/CG 这一套经典 pipeline。

---

四、线性方程组求解与 PCG

牛顿法的每一步，都要解一个（大规模）稀疏线性方程组：

$$
\mathbf{H} \Delta \mathbf{x} = -\mathbf{g},
$$

其中 （几十万甚至上百万）。

求解线性方程组可以分为直接法（由于我们是对称正定矩阵，所以通常会使用 Cholesky 分解：$A = LL^T$）与迭代法（如预处理共轭梯度法 PCG、松弛迭代法如 Gauss-Seidel 或 Jacobi 迭代法）。

$\mathbf{H}$ 的维度等于系统自由度，当系统自由度较少的时候（如布料的网格密度不高，或场景中主要是刚体网格），此时在 CPU 上使用分解法比较高效。

但是直接法的并行度通常较低，不适合 GPU 并行计算，且如果系统自由度较高，则我们很难存储 Cholesky 分解后的矩阵 $L$。因此 LCS 里使用的是基于预处理共轭梯度法（Preconditioned Conjugate Gradient，PCG）的迭代法，它属于 Krylov 子空间方法的一类。

简单回顾一下 PCG 的核心：
- 初始给一个方向 $\mathbf{p}_0$，残差 $\mathbf{r}_0 = \mathbf{b} - \mathbf{A}\mathbf{x}_0$
- 每次迭代在 Krylov 子空间里找一个新的搜索方向 $\mathbf{p}_{k+1}$
- $\mathbf{p}_k$ 和 $\mathbf{p}_{k+1}$ 之间保持 $\mathbf{A}$-正交
- 通过预处理矩阵 $\mathbf{M}^{-1}$ 把问题变成 $\mathbf{M}^{-1}\mathbf{A}$ 更好解的形式

理论上，如果预处理矩阵 $\mathbf{M}$ 等于 $\mathbf{A}^{-1}$，那只要一步就能收敛。但显然这相当于先把问题解完再求解一遍……所以实际预处理只能是一个“近似”，比如：
- 对角预处理（Jacobi）
- 分块对角预处理
- 或更高级的 AMG/ILU 等。

在实际的仿真场景里，线性系统自由度非常高（例如 10 万顶点的布料，自由度 30 万），每个牛顿迭代里，PCG 常常需要几百步甚至上千次迭代，PCG 本身往往是整个 pipeline 里最耗时的部分。在 GPU 上实现 PCG，最核心的算子有两个：
- 稀疏矩阵-向量乘法（SpMV，Sparse Matrix-Vector Multiply）
- 各种归约（内积、范数计算等）

我们的 SpMV 方案参考了 TOG 2025 的论文 [StiffGIPC: Advancing GPU IPC for Stiff Affine-Deformable Simulation](https://dl.acm.org/doi/full/10.1145/3735126)，该论文也是另一个开源仿真项目 [libuipc](https://github.com/spiriMirror/libuipc) 的理论基础。LCS 大量参考了 uipc 的实现与设计，非常感谢 uipc 社区的支持！uipc 是非常成熟的 IPC 仿真框架，也经过了多年的优化，现在已经大量应用到了机器人强化学习任务中。

[通用无穿透物理引擎 LIBUIPC 速览](https://zhuanlan.zhihu.com/p/16559361833)

此外在做 SpMV 前，我们需要将稀疏矩阵中的的相同元素装配到一起（即矩阵装配），这个步骤依赖于排序运算。不过由于高性能的全局基数排序实现起来比较复杂，所以 LCS 目前是在 Block-Level 做了两次排序与累加，因此会有大量的重复元素。

好在 Ligo 老师正在维护一个基于 LuisaCompute 的 通用并行原语库 [lc_parallel_primitive](https://github.com/Ligo04/lc_parallel_primitive)：

[基于 LuisaCompute 的 GPU 并行原语库: LC Parallel Primitive](https://zhuanlan.zhihu.com/p/2012501624061441241)

LCPP 目前已经在多端（CUDA、Metal、DX、Vulkan 等）实现了高性能的排序、规约、扫描、ReduceByKey 等算法（比如基数排序使用了目前最先进的 Decoupled-Lookback 算法，避免了频繁启动小 Kernel 去拿前面 Block 的中间结果），因此我们后续会用 LCPP 进一步提升 LCS 的性能。

---

五、本构模型：从布料到 Affine Body Dynamics

LCS 目前在布料、软体、刚体方面，使用了几类不同的本构（constitutive）模型，更多细节可以参考项目的 [能量文档](https://github.com/ChengzhuUwU/LuisaComputeSimulator/blob/main/Document/Energies.md)：

### 5.1 布料本构

布料部分实现了经典的质点-弹簧模型，可以参考 2002 年的论文 [Stable but Responsive Cloth](https://dl.acm.org/doi/10.1145/566654.566624)：

$$E = \frac{k}{2} (|p_i - p_j| - L_0)^2$$

以及线弹性有限元模型，参考的是 2019 年的 [A Finite Element Formulation of Baraff-Witkin Cloth](https://www.tkim.graphics/FEMBW/)，该论文将 1998 年的经典论文 [Large Steps in Cloth Simulation](https://zhuanlan.zhihu.com/p/449823510) 转化成了有限元的形式，并对能量做了特征值分析，提出了高效计算恒半正定的 Hessian 的计算流程（论文后面还有 Eigen 的源代码，基本上可以直接复制粘贴）。

$$E = A\left(E_{stretch} + E_{shear}\right)$$

$$E_{stretch} = \frac{\mu}{2}\left[(\|F_u\|-1)^2 + (\|F_v\|-1)^2\right]$$

$$E_{shear} = \frac{\lambda}{2}(F_u\cdot F_v)^2$$

这也是我最喜欢的论文。

### 5.2 软体本构：ARAP 与 Stable Neo-Hookean

软体仿真部分，除了直接复用上面的弹簧能量外，我们还实现了 Stable Neo-Hookean 与 ARAP 能量


$$E_{SNHK} = V\left[\frac{\mu}{2}(\mathrm{tr}(F^T F) - 3) - \mu(\det(F)-1) + \frac{\lambda}{2}(\det(F)-1)^2\right]$$

$$E_{ARAP} = \mu V \|F - R\|_F^2$$

> （其实 SNHK 的 Hessian 还有问题，一跑就炸，还在调x）

关于这两个能量的特征值分析，感兴趣的读者可以阅读 [Dynamic Deformables](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/)（其实和上面的布料线弹性有限元模型是一个作者）。


### 5.3 刚体本构：Affine Body Dynamics（ABD）

刚体部分，我们使用的是 2022 年的 Affine Body Dynamics（ABD） 的方法。

传统刚体通常用 6 个自由度（3 平移 + 3 旋转参数）来描述，而 ABD 用一个 3×3 的线性变换矩阵 $\mathbf{A} = \mathbf{RS}$ 加上一个平移 $\mathbf{p}$，总共 12 个自由度来描述刚体：

$$
x = \mathbf{A} \, \bar{x} + \mathbf{p},
$$

其中 $\bar{x}$ 是刚体中的顶点在参考构型下的位置，$x$ 是经过仿射（Affine）变换后的位置。

这样做有几个好处：
1. **消除旋转参数化的奇异性**：不再纠结欧拉角万向节锁、四元数归一化等问题，$\mathbf{A}$ 只是一个线性算子
2. **方便做线性 CCD**：顶点轨迹在一步时间里是仿射的，$\mathbf{x}^{n+1}(t)$ 随时间 $t$ 线性或分段线性变化，这对连续碰撞检测（CCD）非常友好，可以在参数空间上做线性/多项式求根。如果我们对三自由度的旋转量做差值，我们实际上得到的一个非线性路径，TOG 2021 有一篇论文 [Intersection-free Rigid Body Dynamics](https://ipc-sim.github.io/rigid-ipc/) 提出用分段线性路径来近似，处理起来还是比较麻烦
3. **方便和软体管线做耦合**：刚体的能量形式类似一个“刚度很大的软体”，只是它的自由度不再是每个顶点，而是刚体的 12D 仿射参数
4. **允许轻微的弹性形变**：如果刚度设置得非常大，理想刚体会保持 $\mathbf{A}$ 接近正交矩阵；但 ABD 允许一定的拉伸 / 压缩，这在真实材料里反而更合理，也有助于和有限元软体做统一处理。

在处理涉及刚体顶点的能量（如碰撞能量）的时候，需要遵循链式法则：

$$ \frac{\partial E}{\partial q} = \frac{\partial E}{\partial x} \frac{\partial x}{\partial q} = \frac{\partial E}{\partial x} J$$

$$ \frac{\partial^2 E}{\partial q_i \partial q_j} 
= (\frac{\partial x}{\partial q_j})^T  \frac{\partial^2 E}{\partial x^2} \frac{\partial x}{\partial q_i} + \cancel{\frac{\partial E}{\partial x} \frac{\partial^2 x}{\partial q_i \partial q_j}} 
= J_j^T \frac{\partial^2 E}{\partial x_i \partial x_j} J_i$$

其中 Jacobian 矩阵 $J$ 描述来顶点坐标 $x$ 与刚体状态 $q$ 的关系。具体形式是：

$$ J = \frac{\partial x}{\partial q} = 
\begin{bmatrix}
1 & 0 & 0 & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 & & & & & & \\
0 & 1 & 0 & & & & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 & & & \\
0 & 0 & 1 & & & & & & & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 \\
\end{bmatrix} \in R^{3 \times 12} $$


---

六、碰撞：DCD / CCD、能量形式与 Hessian 装配

碰撞检测和响应是整个管线中最复杂、也是最能拉开 GPU / CPU 差距的一环。

### 6.1 离散碰撞检测（Discrete Collision Detection，DCD）

在 DCD 阶段（也就是“几何检测”），我们主要做两类基本原语的距离检测：
- VF：顶点-三角形（Vertex-Face）
- EE：边-边（Edge-Edge）

这两类已经能覆盖大多数三维网格之间的接触情况。实际实现时会遇到退化情形：
- VF 退化成 VV（点-点）
- EE 退化成 VE（点-边）

在 IPC 体系中，通常会把它们区分开来；但在 LuisaComputeSimulator 中，我们采用了一个更统一的做法：始终以 VF / EE 的形式存储一条“碰撞约束”。如果几何上退化成了 VV/VE，就把对应顶点的重心坐标直接设为 0。

在一次牛顿迭代中，我们把这些重心坐标视为常量（Gauss-Newton 风格的近似），这样距离对各个顶点自由度的偏导只依赖于重心坐标与接触方向，而不显式区分图元的具体类型。这种 Gauss-Newton 的表示参考的是 SIGGRAPH 2024 论文 [Preconditioned Nonlinear Conjugate Gradient Method for Real-time Interior-point Hyperelasticity]( https://xingbaji.github.io/PNCG_project_page/)。


### 6.2 连续碰撞检测（Continuous Collision Detection， CCD）：C-IPC 风格的流程

如果只做 DCD，很容易出现 **隧穿**：即物体在单步内移动距离大于安全距离 $\hat d$，CCD 会检测运动轨迹中是否发生穿透，如果穿透则需要回退到无穿透到位置。

我们使用的是 IPC 论文的 Global CCD。即在全局所有候选碰撞对中选取最小的碰撞时间（Time of Impact，TOI），然后以这个最小的 TOI 为准回退整体步长。这种思路是最符合运动过程的，不过收敛较慢。有一些模拟系统会在 Global CCD 和 Local CCD（逐顶点 TOI）之间做权衡。

我们的 DCD（离散位置检测）和 CCD 的实现参考了 [ZOZO's Contact Solver](https://github.com/st-tech/ppf-contact-solver) 项目，这个项目是 TOG 2024 论文 [A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness](https://dl.acm.org/doi/abs/10.1145/3687908) 的源码。其中 CCD 的原理是 SIGGRAPH 2021 论文 [Codimensional Incremental Potential Contact (C-IPC)](https://ipc-sim.github.io/C-IPC/) 中的 Adaptive CCD，是目前最先进的连续碰撞检测方案，兼容厚度的检测。

### 6.3 碰撞能量的形式：Quadratic 与 Log-Barrier

在物理仿真中，碰撞处理方式通常是给空间上邻近的图元施加一个虚拟的弹簧：设 $d$ 是几何距离，$\hat d$ 是安全距离（通常在 1–20 mm 之间），定义约束：

$$
C = d - \hat d > 0
$$

常见的一些能量形式包括：
1. 二次型能量（Quadratic）：$E_{\text{quad}} = \frac{1}{2} \kappa C^2$
2. 三次型能量（Cubic）：$E_{\text{quad}} = \frac{1}{2} \kappa C^3$
3. 对数形式能量（Log-Barrier）：$-\log(d / \hat d)$

其中 $\kappa$ 是劲度系数，通常在 $10^6 \sim 10^{10}$ 的量级。

二次型能量形式比较简单，导数也非常好算，但是提供的最大斥力有限，数量级大约是 $\kappa \hat d$。

三次型能量或者更高次的多项式能量在靠近 $d = 0$ 的时候提高增长速度，但提供的斥力仍然是有限的，具体可以参考 A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness 中的论述，需要与动态调节劲度系数 $\kappa$ 相结合。

对数形式能量在 $d \to 0$ 时会趋向 $+\infty$，理论上可以提供“无上限”的斥力。不过该能量在 $d = \hat d$ 处不够光滑。

因此我们使用的是 IPC 中的平滑形式对数能量（Smoothed Log-Barrier）：

$$
E_{\text{IPC}} = -\kappa (d - \hat d)^2 \log\left(\frac{d}{\hat d}\right).
$$

这个形式有几个非常关键的好处：
- 当 $d \to 0$ 时，对数项让能量趋向无穷大，可以提供“无限大”的理论斥力
- $d \to \hat d$ 时，前面的 $(d - \hat d)^2$ 会把能量、梯度、Hessian 都平滑到 0
- 整个能量关于距离是 $C^2$ 连续的，非常适合牛顿迭代

不过，需要强调的是：“无限大”在实现中并不是真的无限：当 $d < 10^{-5}\,\text{m}$（0.01mm 量级）时，浮点精度已经很难可靠地区分更小的距离。即使能量公式写着 $+\infty$，实际算出来也会因为 underflow / overflow 或精度误差而失真

所以实践中往往仍然需要配合调整 $\kappa$，调整的依据通常基于场景中的最大外力。


七、LuisaCompute 部分：Dynamic Resize 与多线程 JIT

### 7.1 Dynamic Resize：动态缓冲与 BindingGroup

碰撞相关的数据结构的数量是动态变化的，很难一开始就预估一个“小而刚好”的上界。典型的例子包括：
- BroadPhase（宽检测阶段）之后的候选碰撞对（通过 BVH AABB 剔除得到）
- NarrowPhase（窄检测阶段）几何检测之后的 VF / EE 碰撞对
- 如果扩展到 MPM / 弹性杆 仿真，还会有 VV / VE 碰撞对
- 装配后的 Hessian triplet 列表

直觉上我们当然可以给每个 Buffer 分配一个巨大的空间，但这在存储层面都非常浪费。所以实际做法是如果发现超过了 Buffer 当前容量，则需要做动态 Resize。但这会带来一个问题：如果 Kernel 里是以 捕获 BufferView 对方式访问内存，则在 Resize 之后，原来被捕获的指针就失效了。

一个方案是将这种动态变化的 Buffer 改用参数传递（类似于 CUDA 传入指针的方式），但是写起来比较麻烦。另一个方案是使用 LuisaCompute 的 BindingGroup：BindingGroup 允许你在一个结构体里访问到多个 Buffer，本质上还是逐个绑定，所以不用做额外的 Shader Reload 这样的操作。

而且 BindingGroup 非常适合做数据分类与解耦，后面我将系统中的材料能量（比如惯性势能，拉伸能量，弯曲能量）也都替换成了这种 BindingGroup 的形式。

> 不过需要注意的是，因为 BindingGroup 会一次性绑定其中的多个 Buffer，在某些后端（例如 DX）上会受到 绑定参数总大小 的限制（例如总指针大小不能超过 64 Byte），所以在设计 BindingGroup 结构时，也要兼顾这些硬件约束

### 7.3 多线程 JIT：80+ Kernels 的冷启动问题

物理仿真涉及非常多的 kernel：
- 各种能量项的 Gradient / Hessian 计算
- 碰撞检测（BroadPhase / NarrowPhase / CCD）
- 各种 PCG/线性代数算子
- 以及一堆辅助工具 Kernel

项目里 Kernel 总数在 80 个以上，如果全部在第一次运行时串行 JIT 编译，冷启动体验非常糟糕，尤其在 CUDA 后端（NVCC 本身就比较慢），等几分钟都不稀奇。为了解决这个问题，麦老师@ 帮忙写了一套多线程 JIT 流程，在程序启动时，把所有 Kernel 的编译任务分发到多个线程，对于 DX / Metal 这类后端，整体 JIT 时间可以压到 1s 左右，几乎是“秒开”，CUDA 后段也可以压缩到十几秒（在 LLVM CodeGen 下会进一步压缩）

> 在多线程环境下，如果在 Kernel 中使用 引用捕获的 Lambda 表达式 述，很容易因为引用捕获的上下文已经失效 / 被移动而出现错误，更安全的做法是统一改成值捕获
> 
> 长期目标的话，我个人还是比较倾向于 AOT 的方案

### 7.4 Cmake & Xmake

项目支持 Xmake 编译，感谢 @ligo 与 @星姐！不过最近我改动的有点多，自己的电脑 Xmake 环境烂了，所以有段时间没有维护，欢迎大家来 PR！

---

结语

最近两年没有在知乎上写东西了，似乎一直在忙论文。。。（虽然忙了这么久只有一篇），而我的工作效率也很低，经常需要对着电脑发呆半天或者发呆几天，才能把一个细节想清楚。

这是我非常重要的项目，事实上我现在科研的项目正是 Fork 于某个 Commit 节点，我真切希望这个社区可以越来越好。后面还会继续填坑（比如更好的预条件子、更高阶的能量模型、LCPP 加速的装配、关节约束、弹性杆、体网格生成等等），也欢迎感兴趣的同学一起交流、提 issue 与 PR。

如果你也有“写一套自己的物理引擎 / GPU 计算框架”的想法，希望这篇文章能给你一点点启发。


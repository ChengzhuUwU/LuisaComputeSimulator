# LuisaComputeSolver

![Teasor](Resources/OutputImage/README1.png)

## How to start

1. Clone the repository: `git clone https://github.com/ChengzhuUwU/libAtsSim.git --recursive`

2. Download required packages: (`brew install` or `vcpkg install`) ` Eigen3, tbb, glfw3`
    - (For windows user: You may need to set `CMAKE_PREFIX_PATH` in 'CmakeLists.txt' with your cmake path)

3. Configure: `cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++`

> Or you can specify the compiler using `-DCMAKE_C_COMPILER=/usr/bin/clang`

4. Build: `cmake --build build -j`

4. Run: `build/bin/app-simulation` or `build/bin/app-simulation.exe`

> Recommend Compiler: [Clang 15~18](https://github.com/llvm/llvm-project/releases/tag/llvmorg-18.1.8)
>
> Recommend Generator: ninja (Use [pre-build](https://github.com/ninja-build/ninja/releases/tag/v1.13.1) or [build from source](https://github.com/ninja-build/ninja))

## Reference

IPC: libuipc、solid-sim-toturial

PCG: MAS

LBVH: Document by suika, libuipc

Device Intrinsic: LuisaComputeGaussSplatting


## 其他

项目还是非常早期的阶段，仅供学习与玩一玩，更完整、严谨的仿真管线可以参考 [libuipc](https://github.com/spiriMirror/libuipc) 与 minchen 的 [Totorial](https://github.com/phys-sim-book/solid-sim-tutorial)，希望下一个工作早点中 TT，就能补一些东西了

为什么想做这个：狗蛋、minchen、mike、anka、suika、zihang、kemeng、yupeng、xinlei、chenjiong 等老师的开源之光太耀眼了

LuisaCompute 的优势：写起来很方便，不用天天bind了，slang也难逃此劫。。虽然默认是jit，但感觉改成aot问题也不大，可以用于生产中。

LuisaCompute 的劣势：什么都得自己写，特别是线性代数部分

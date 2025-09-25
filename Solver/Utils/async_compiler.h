#pragma once
#include <luisa/runtime/device.h>
#include <luisa/core/fiber.h>
#include <luisa/dsl/func.h>

namespace lcs {
class AsyncCompiler {
    luisa::fiber::counter _counter;
    luisa::compute::Device _device;
public:
    auto &device() { return _device; }
    auto const &device() const { return _device; }
    AsyncCompiler(luisa::compute::Device &device) : _device(device) {}
    AsyncCompiler(AsyncCompiler const &) = delete;
    AsyncCompiler(AsyncCompiler &&) = default;
    ~AsyncCompiler() = default;
    template<size_t N, typename Func, typename... Args>
    void compile(luisa::compute::Shader<N, Args...> &result,
                               Func const &kernel,
                               const luisa::compute::ShaderOption &option = {}) noexcept {
        _counter.add();
        luisa::fiber::schedule([option, &result, kernel, counter = this->_counter, device = this->_device]() mutable {
            result = device.compile<N>(kernel, option);
            counter.done();
        });
        // _counter.wait();
        // result = this->_device.compile<N>(kernel, option);
    }
    // template<size_t N, typename Func, typename Shader>
    //     requires(std::negation_v<luisa::compute::detail::is_dsl_kernel<std::remove_cvref_t<Func>>> && N >= 1 && N <= 3)
    // [[nodiscard]] void compile(Shader &result, Func &&f, const luisa::compute::ShaderOption &option = {}) noexcept {
    //     _counter.add();
    //     luisa::fiber::schedule([option, &result, &f, counter = this->_counter, device = this->_device]() mutable {
    //         result = device.compile(f, option);
    //         counter.done();
    //     });
    // }
    void wait() {
        _counter.wait();
    }
};
}// namespace lcs
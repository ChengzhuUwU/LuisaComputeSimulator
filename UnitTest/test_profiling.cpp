// 请你做下面三件事情：
// 1. 读取分支 lcpp 的提交，将 lcpp 集成到当前的 dev 分支中
// 2.

#include <luisa/luisa-compute.h>

int main(int argc, char** argv)
{
	luisa::log_level_info();
	LUISA_INFO("Test LBVH refit");

	// Init GPU system
#if defined(__APPLE__)
	std::string backend = "metal";
#else
	std::string backend = "cuda";
#endif
	const std::string			 binary_path(argv[0]);
	luisa::compute::Context		 context{ binary_path };
	luisa::vector<luisa::string> device_names = context.backend_device_names(backend);
	if (device_names.empty())
	{
		LUISA_WARNING("No hardware device found.");
		exit(1);
	}
	if (argc >= 2)
	{
		backend = argv[1];
	}
	luisa::compute::Device device = context.create_device(backend, nullptr, true);
	luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

	using namespace luisa::compute;

	auto fn_update_vert_tree_leave_aabb = device.compile<1>(
		[](const Var<luisa::compute::BufferView<float4x4>> buffer, const Var<uint> loop_time)
		{
			const auto vid = dispatch_id().x;
			auto	   value = buffer->read(vid);
			$for(i, loop_time)
			{
				value = value * value;
			};
			buffer->write(vid, value);
		});

	auto fn_empty_kernel = device.compile<1>(
		[]()
		{
			const auto vid = dispatch_id().x;
		});

	const uint			  test_times = 100;
	const uint			  buffer_size = 100000;
	Buffer<float4x4>	  test_buffer = device.create_buffer<float4x4>(buffer_size);
	std::vector<float4x4> host_buffer(buffer_size, float4x4::eye(1.0f));
	stream << test_buffer.copy_from(host_buffer.data()) << synchronize();

	std::vector<luisa::Clock> clocks(test_times);
	std::vector<double>		  durations(test_times);

	auto fn_test_compile_refit = [&](const uint i, const uint loop_time)
	{
		stream << fn_empty_kernel().dispatch(1);
		stream << [&]()
		{
			clocks[i].tic();
		};
		for (uint j = 0; j < 10; j++)
			stream << fn_update_vert_tree_leave_aabb(test_buffer.view(), loop_time).dispatch(buffer_size);
		stream << synchronize() << [&]()
		{
			durations[i] = clocks[i].toc();
		};
		stream << fn_empty_kernel().dispatch(1);
		stream << synchronize();
	};

	for (uint i = 0; i < test_times; i++)
	{
		const uint loop_time = i * 100;
		fn_test_compile_refit(i, loop_time);
		LUISA_INFO("Loop {}, Loop time: {}, duration: {:.4f} ms", i, loop_time, durations[i]);
	}
	// for (uint i = 0; i < test_times; i++)
	// {
	// 	LUISA_INFO("Loop time: {}, duration: {} ms", i + 1, durations[i]);
	// }

	return 0;
}
#include <luisa/luisa-compute.h>
int main(int argc, char** argv)
{
	// Initialize devices
	luisa::compute::Context context{ argv[0] };
	luisa::compute::Device	device = context.create_device("cuda" /*or: dx, metal, vk, fallback(CPU)*/, nullptr, false);
	luisa::compute::Stream	stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

	uint buffer_size = 1000;
	// Initialize resources
	luisa::compute::Buffer<float> buffer_in1;
	luisa::compute::Buffer<float> buffer_in2;
	luisa::compute::Buffer<float> buffer_out;
	buffer_in1 = device.create_buffer<float>(buffer_size);
	buffer_in2 = device.create_buffer<float>(buffer_size);
	buffer_out = device.create_buffer<float>(buffer_size);

	// Data upload
	std::vector<float> host_vector_1(buffer_size, 1.0f);
	std::vector<float> host_vector_2(buffer_size, 2.0f);
	stream << buffer_in1.copy_from(host_vector_1.data())
		   << buffer_in2.copy_from(host_vector_2.data())
		   << luisa::compute::synchronize(); // Optional synchronization

	// Kernel implemtation
	luisa::compute::Shader<1, float> fn_add_float = device.compile<1>(
		[&](const luisa::compute::Var<float> other_params)
		{
			using Index = luisa::compute::Var<uint>;
			using Real = luisa::compute::Var<float>;
			Index index = luisa::compute::dispatch_id().x;
			Real  input_val1 = buffer_in1->read(index);
			Real  input_val2 = buffer_in2->read(index);
			Real  result = input_val1 + input_val2;
			buffer_out->write(index, result);
		});

	// Launch kernel
	stream << fn_add_float(0.0f).dispatch(buffer_size);

	// Data download
	std::vector<float> host_vector_3(buffer_size, 1.0f);
	stream << buffer_out.copy_to(host_vector_3.data())
		   << luisa::compute::synchronize();
}
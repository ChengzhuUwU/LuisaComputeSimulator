#include "lcpp/device/device_radix_sort.h"
#include "luisa/core/logging.h"
#include "luisa/runtime/buffer.h"
#include <algorithm>
#include <functional>
#include <lcpp/parallel_primitive.h>
#include <string_view>
#include <vector>
using namespace luisa::parallel_primitive;

template <typename T, typename FnGetDesired>
bool check_template(const T& actual, FnGetDesired fn_get_desired, const std::string_view test_name, uint size)
{
	for (uint i = 0; i < actual.size(); ++i)
	{
		auto desired = fn_get_desired(i);
		if (actual[i] != desired)
		{
			// LUISA_WARNING("Test failed at size {:5}: index {}, expected {}, got {}", size, i, desired, actual[i]);
			LUISA_WARNING("{:12} {:4} elements: Failed at index {}, expected {}, got {}", test_name, size, i, desired, actual[i]);
			if (actual.size() <= 64)
			{
				LUISA_WARNING("{:12} {:4} elements: actual: {}", test_name, size, actual);
			}
			return false;
		}
	}
	LUISA_INFO("{:12} {:4} elements: all values are correct.", test_name, size);
	return true;
}

// Reduce
void test_device_reduce(Device& device, Stream& stream)
{
	DeviceReduce<> device_reduce;
	device_reduce.create(device);

	using Type4Byte = int;

	for (uint loop = 0; loop < 24; ++loop)
	{
		uint				   num_items = 1 << loop;
		Buffer<Type4Byte>	   d_input = device.create_buffer<Type4Byte>(num_items);
		Buffer<Type4Byte>	   d_output = device.create_buffer<Type4Byte>(1);
		size_t				   temp_bytes = DeviceReduce<>::GetTempStorageBytes<Type4Byte>(num_items);
		Buffer<uint>		   d_temp = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));
		std::vector<Type4Byte> host_input(num_items, 1);
		stream << d_input.copy_from(host_input.data()) << synchronize();
		CommandList cmdlist;
		device_reduce.Sum(cmdlist, d_temp.view(), d_input.view(), d_output.view(), num_items);
		stream << cmdlist.commit() << synchronize();

		std::vector<Type4Byte> host_output(1);
		stream << d_output.copy_to(host_output.data()) << synchronize();

		check_template(
			host_output, [num_items](uint) -> Type4Byte
			{ return num_items; },
			"Device Reduce", num_items);

		d_input.release();
		d_output.release();
		d_temp.release();
	}
}

// Scan (Prefix sum)
void test_device_scan(Device& device, Stream& stream)
{
	DeviceScan<> device_scan;
	device_scan.create(device);
	for (uint loop = 0; loop < 24; ++loop)
	{
		uint			  num_items = 1 << loop;
		Buffer<uint>	  d_keys_in = device.create_buffer<uint>(num_items);
		Buffer<uint>	  d_keys_out = device.create_buffer<uint>(num_items);
		size_t			  temp_bytes = DeviceScan<>::GetTempStorageBytes<uint>(num_items);
		Buffer<uint>	  d_temp = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));
		std::vector<uint> host_keys(num_items);
		for (uint i = 0; i < num_items; ++i)
		{
			host_keys[i] = 1;
		}
		stream << d_keys_in.copy_from(host_keys.data()) << synchronize();
		CommandList cmdlist;
		device_scan.ExclusiveSum(cmdlist, d_temp.view(), d_keys_in.view(), d_keys_out.view(), num_items);
		stream << cmdlist.commit() << synchronize();

		std::vector<uint> host_keys_out(num_items);
		stream << d_keys_out.copy_to(host_keys_out.data()) << synchronize();

		d_keys_in.release();
		d_keys_out.release();
		d_temp.release();

		check_template(
			host_keys_out, [](uint i) -> uint
			{ return i; },
			"Device Scan", num_items);
	}
}

// Radix Sort
void test_device_radix_sort(Device& device, Stream& stream)
{
	DeviceRadixSort<> device_radix_sort;
	device_radix_sort.create(device);
	for (uint loop = 1; loop < 20; ++loop)
	{
		uint			  num_items = 1 << loop;
		Buffer<uint>	  d_keys_in = device.create_buffer<uint>(num_items);
		Buffer<uint>	  d_keys_out = device.create_buffer<uint>(num_items);
		size_t			  temp_bytes = DeviceRadixSort<>::GetSortKeysTempStorageBytes<uint>(num_items);
		Buffer<uint>	  d_temp = device.create_buffer<uint>(bytes_to_uint_count(temp_bytes));
		std::vector<uint> host_keys(num_items);
		for (uint i = 0; i < num_items; ++i)
		{
			host_keys[i] = num_items - i - 1;
		}
		stream << d_keys_in.copy_from(host_keys.data()) << synchronize();
		CommandList cmdlist;
		device_radix_sort.SortKeys<uint>(cmdlist, d_temp.view(), d_keys_in.view(), d_keys_out.view(), num_items);
		stream << cmdlist.commit() << synchronize();

		std::vector<uint> host_keys_out(num_items);
		stream << d_keys_out.copy_to(host_keys_out.data()) << synchronize();

		d_keys_in.release();
		d_keys_out.release();
		d_temp.release();

		check_template(
			host_keys_out, [](uint i) -> uint
			{ return i; },
			"Device Radix Sort", num_items);
	}
}

int main(int argc, char* argv[])
{
	Context ctx{ argv[0] };
	auto	backend = argc > 1 ? argv[1] : "metal";
	Device	device = ctx.create_device(backend);
	Stream	stream = device.create_stream();

	test_device_reduce(device, stream);
	test_device_scan(device, stream);
	test_device_radix_sort(device, stream);
	return 0;
}
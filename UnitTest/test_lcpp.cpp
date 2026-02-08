#include <lcpp/parallel_primitive.h>
#include <vector>
using namespace luisa::parallel_primitive;
int main(int argc, char* argv[])
{

    // Create device and stream
    Context ctx{argv[0]};
    Device  device = ctx.create_device("cuda");
    Stream  stream = device.create_stream();

    // Device-level reduce example
    DeviceReduce<> device_reduce;
    device_reduce.create(device);


    Buffer<int> input  = device.create_buffer<int>(1024);
    Buffer<int> output = device.create_buffer<int>(1);

    std::vector<int> host_input(1024, 1);
    stream << input.copy_from(host_input.data()) << synchronize();

    CommandList cmdlist;
    device_reduce.Sum(cmdlist, stream, input.view(), output.view(), 1024);
    stream << cmdlist.commit() << synchronize();

    std::vector<int> host_output(1);
    stream << output.copy_to(host_output.data()) << synchronize();
    LUISA_INFO("Sum result: {}", host_output[0]);
}
#include "Core/xbasic_types.h"
#include <iostream>
#include <luisa/luisa-compute.h>
#include <Eigen/Dense>

int main(int argc, char** argv)
{
    luisa::log_level_info();
    std::cout << "Hello, LuisaComputeSimulation!" << std::endl;
    
    // Init GPU system
#if defined(__APPLE__)
    std::string    backend          = "metal";
#else
    std::string    backend          = "cuda";
#endif
    const std::string binary_path(argv[0]);
    luisa::compute::Context context{ binary_path };
    luisa::vector<luisa::string> device_names = context.backend_device_names(backend);
    if (device_names.empty()) { LUISA_WARNING("No haredware device found."); exit(1); }
    for (size_t i = 0; i < device_names.size(); ++i) { LUISA_INFO("Device {}: {}", i, device_names[i]); }
    luisa::compute::Device device = context.create_device(backend);
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    // Outer product
    if (false)
    {
        lcs::VECTOR12 vec1;
        lcs::VECTOR12 vec2;
        lcs::set_largevec(vec1, 1.0f);
        lcs::set_largevec(vec2, 1.0f);
        vec1[1] = luisa::make_float3(0);

        auto large_mat = lcs::outer_product_largevec(vec1, vec2);
        lcs::print_largemat(large_mat);
    }
    // if (false)
    {
        lcs::VECTOR12 vec1; lcs::set_largevec(vec1, 1.0f);
        lcs::MATRIX12 mat1; lcs::set_largemat_identity(mat1); lcs::set_colomn_largemat(mat1, 0, vec1);
        lcs::MATRIX12 mat2; lcs::set_largemat_identity(mat2);

        lcs::mult_largemat_scalar(mat1, mat1, 2.0f);
        auto result = lcs::add_largemat(mat1, mat2);
        lcs::print_largemat(result);
    }
    if (false)
    {
        {
            Eigen::Vector<float, 12> vec1; vec1.setOnes(); vec1 << 1,2,3,4,5,6,7,8,9,10,11,12;
            Eigen::Vector<float, 12> vec2; vec2.setOnes(); vec2 *= 2.0f;
            Eigen::Matrix<float, 12, 12> mat1; mat1.setIdentity(); mat1.col(0) = vec1;
            Eigen::Matrix<float, 12, 12> mat2; mat2.setIdentity(); mat2.row(0) = vec2;
            auto result1 = mat1 * mat2;
            auto result2 = result1 * vec1;
            // std::cout << mat1 << std::endl;
            // std::cout << mat2 << std::endl;
            std::cout << result1 << std::endl;
            std::cout << result2 << std::endl;
        }
        lcs::VECTOR12 vec1; lcs::set_largevec(vec1, 1.0f); vec1.vec[0] = luisa::make_float3(1,2,3); vec1.vec[1] = luisa::make_float3(4,5,6); vec1.vec[2] = luisa::make_float3(7,8,9); vec1.vec[3] = luisa::make_float3(10,11,12);
        lcs::VECTOR12 vec2; lcs::set_largevec(vec2, 2.0f);
        lcs::MATRIX12 mat1; lcs::set_largemat_identity(mat1); lcs::set_colomn_largemat(mat1, 0, vec1);
        lcs::MATRIX12 mat2; lcs::set_largemat_identity(mat2); lcs::set_row_largemat(mat2, 0, vec2);

        auto result1 = lcs::mult_largemat_mat(mat1, mat2);
        auto result2 = lcs::mult_largemat_vec(result1, vec1);
        // lcs::print_largemat(mat1);
        // LUISA_INFO("");
        // lcs::print_largemat(mat2);
        lcs::print_largemat(result1);
        lcs::print_largevec(result2);


    }

    // lcs::set_colomn_largemat(mat, 0, vec);
    // lcs::set_row_largemat(mat, 0, vec);
    
    // lcs::mult_largemat(mat, mat, 3.0f);
    // // lcs::print_largemat(mat); 
    
    // mat = lcs::mult_largemat(mat, 1.25f);
    // lcs::print_largemat(mat);

    return 0;
}
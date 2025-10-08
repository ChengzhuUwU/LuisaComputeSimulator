function(luisa_compute_solver_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE luisa::compute luisa-compute-solver-lib)
    install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()



luisa_compute_solver_add_test(test-xtypes test_xtypes.cpp)
luisa_compute_solver_add_test(test-lc-features test_lc_features.cpp)
luisa_compute_solver_add_test(test-lbvh-refit test_lbvh_refit.cpp)


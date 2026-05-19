function(luisa_compute_solver_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE luisa::compute luisa-compute-solver-lib)
    install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()



luisa_compute_solver_add_test(test_xtypes test_xtypes.cpp)
luisa_compute_solver_add_test(test_lc_features test_lc_features.cpp)
luisa_compute_solver_add_test(test_lbvh_refit test_lbvh_refit.cpp)
luisa_compute_solver_add_test(test_jit test_jit.cpp)
luisa_compute_solver_add_test(test_host_functions test_host_functions.cpp)
luisa_compute_solver_add_test(test_abd_energy test_abd_energy.cpp)
luisa_compute_solver_add_test(test_gradient_hessian test_gradient_hessian.cpp)
luisa_compute_solver_add_test(test_newton_solver_integration test_newton_solver_integration.cpp)
luisa_compute_solver_add_test(test_profiling test_profiling.cpp)
luisa_compute_solver_add_test(test_lcpp test_lcpp.cpp)

# -----------------------------------------------------------------------------
# New modular unit tests (IPC framework testing suite)
# -----------------------------------------------------------------------------
luisa_compute_solver_add_test(test_lbvh               test_lbvh.cpp)
luisa_compute_solver_add_test(test_narrow_phase       test_narrow_phase.cpp)
luisa_compute_solver_add_test(test_energy_assembly     test_energy_assembly.cpp)
luisa_compute_solver_add_test(test_pcg_solver         test_pcg_solver.cpp)
luisa_compute_solver_add_test(test_ccd                 test_ccd.cpp)
luisa_compute_solver_add_test(test_integration         test_integration.cpp)

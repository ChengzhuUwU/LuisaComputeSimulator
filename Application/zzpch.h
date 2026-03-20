#include <Eigen/Dense>
#if defined(SIMULATION_APP_USE_GUI)
	#include <polyscope/polyscope.h>
	#include <polyscope/surface_mesh.h>
	#include <imgui.h>
#endif
#include <filesystem>
#include <istream>
#include <sstream>
#include <vector>
#include <string>
#define LUISA_ENABLE_DSL
#include <luisa/luisa-compute.h>
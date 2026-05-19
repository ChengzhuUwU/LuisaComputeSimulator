#!/bin/bash
# =============================================================================
# run_tests.sh - Run all IPC framework unit tests
# Usage: ./run_tests.sh [test_name] [--all] [--summary] [--nobuild]
# Requirements: bash 4.0+ on macOS/Linux
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/../build}"
TEST_BIN_DIR="${BUILD_DIR}/bin"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║          IPC Physics Simulation - Unit Test Runner                   ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Detect build system
if [ -d "${BUILD_DIR}" ]; then
    if [ -f "${BUILD_DIR}/build.ninja" ] || [ -d "${BUILD_DIR}/build" ]; then
        BUILD_SYSTEM="xmake"
    elif [ -f "${BUILD_DIR}/Makefile" ] || [ -f "${BUILD_DIR}/cmake_install.cmake" ]; then
        BUILD_SYSTEM="cmake"
    else
        BUILD_SYSTEM="unknown"
    fi
else
    BUILD_SYSTEM="none"
fi

echo "Build directory: ${BUILD_DIR}"
echo "Build system: ${BUILD_SYSTEM}"
echo ""

# Get CPU count (macOS compatible)
get_cpu_count() {
    sysctl -n hw.ncpu 2>/dev/null || echo "4"
}

# Build tests if needed
if [ "$1" != "--nobuild" ]; then
    echo "Building tests..."
    echo "============================================================"

    if [ "${BUILD_SYSTEM}" = "xmake" ]; then
        cd "${BUILD_DIR}"
        xmake build -j "$(get_cpu_count)" test_lbvh test_narrow_phase test_energy_assembly test_pcg_solver test_ccd test_integration 2>&1 | tail -30
    elif [ "${BUILD_SYSTEM}" = "cmake" ]; then
        cd "${BUILD_DIR}"
        cmake --build . -j "$(get_cpu_count)" 2>&1 | tail -30
    else
        echo -e "${YELLOW}Warning: No build directory found. Please build tests manually.${NC}"
        echo "  cd ${BUILD_DIR} && cmake .. && cmake --build ."
    fi
    echo ""
fi

# List of all tests
ALL_TESTS="test_lbvh test_narrow_phase test_energy_assembly test_pcg_solver test_ccd test_integration"

# Simple test runner using a temp file
TEST_RESULTS_FILE="/tmp/test_results_$$.txt"
> "${TEST_RESULTS_FILE}"

run_test() {
    local test_name="$1"
    local test_bin="${TEST_BIN_DIR}/${test_name}"

    if [ ! -f "${test_bin}" ]; then
        echo -e "${YELLOW}[SKIP]${NC} ${test_name} (binary not found)"
        echo "${test_name}:SKIP" >> "${TEST_RESULTS_FILE}"
        return 2
    fi

    echo -e "${BLUE}Running:${NC} ${test_name}"
    echo "------------------------------------------------------------"

    local start_time=$(date +%s)
    if "${test_bin}" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}[PASS]${NC} ${test_name} (${duration}s)"
        echo "${test_name}:PASS:${duration}" >> "${TEST_RESULTS_FILE}"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${RED}[FAIL]${NC} ${test_name} (${duration}s)"
        echo "${test_name}:FAIL:${duration}" >> "${TEST_RESULTS_FILE}"
        return 1
    fi
}

# Run single test
if [ -n "$1" ] && [ "$1" != "--all" ] && [ "$1" != "--summary" ] && [ "$1" != "--nobuild" ]; then
    run_test "$1"
    rm -f "${TEST_RESULTS_FILE}"
    exit $?
fi

# Run all tests
echo "Running all IPC framework unit tests..."
echo "============================================================"
echo ""

total_passed=0
total_failed=0
total_skipped=0

for test_name in ${ALL_TESTS}; do
    run_test "${test_name}" || true
    echo ""
done

# Summary
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                        Test Summary                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
printf "  %-35s %s\n" "Test" "Result"
printf "  %-35s %s\n" "----" "------"

# Parse results
while IFS=: read -r test result duration; do
    case "${result}" in
        PASS)
            color="${GREEN}"
            ((total_passed++))
            ;;
        FAIL)
            color="${RED}"
            ((total_failed++))
            ;;
        SKIP)
            color="${YELLOW}"
            ((total_skipped++))
            ;;
        *)
            continue
            ;;
    esac
    printf "  %-35s ${color}%s${NC}\n" "${test}" "${result}"
done < "${TEST_RESULTS_FILE}"

total_tests=$((total_passed + total_failed + total_skipped))

echo ""
echo "  Total: ${total_tests} tests"
echo "  Passed: ${total_passed}"
echo "  Failed: ${total_failed}"
echo "  Skipped: ${total_skipped}"
echo ""

rm -f "${TEST_RESULTS_FILE}"

if [ ${total_failed} -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════════╗"
    echo -e "║                    ALL TESTS PASSED                                    ║"
    echo -e "╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${RED}╔═══════════════════════════════════════════════════════════════════════╗"
    echo -e "║                    SOME TESTS FAILED                                    ║"
    echo -e "╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi

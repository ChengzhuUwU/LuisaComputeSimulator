#!/usr/bin/env python3
"""
Shared argument parser + batch test runner for LuisaComputeSimulator Python bindings.

Independent scripts::

    from run_tests import create_default_parser
    parser = create_default_parser()
    parser.add_argument("--my-flag", ...)   # extend with script-specific args
    args = parser.parse_args()

Batch runner::

    python run_tests.py --backend metal --frames 120

CI integration::

    python PythonBindings/tests/run_tests.py --backend metal --frames 120
"""

import os
import sys
import subprocess
import shutil
import time
import platform
import argparse
from pathlib import Path

from utils.test_script_path import PROJECT_ROOT

OUTPUT_ROOT = Path(PROJECT_ROOT) / "output" / "tests"

# ---------------------------------------------------------------------------
# Test inventory (relative to PROJECT_ROOT / "PythonBindings" / "tests")
# ---------------------------------------------------------------------------
TESTS = [
    "test_cloth_animation.py",
    # "test_cloth_garment_animation.py",
    "test_cloth_large_deform.py",
    "test_cloth_soft_rigid_coupling.py",
    "test_load_from_json.py",
    "test_rigid_animation.py",
    "test_rigid_folding.py",
    "test_rigid_joint_animation.py",
    "test_rigid_joint_animation2.py",
    "test_soft_folding.py",
    "test_tet_drop.py",
    "test_tet_unit.py",
]

DEFAULT_BACKEND = "metal" if platform.system() == "Darwin" else "cuda"


# ---------------------------------------------------------------------------
# Shared argument parser factory
# ---------------------------------------------------------------------------

def create_default_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-populated with the standard CLI flags.

    Callers may add further arguments before calling ``parser.parse_args()``.
    """
    parser = argparse.ArgumentParser(description="LuisaCompute Python example")
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=["cuda", "dx", "metal", "vk", "fallback", "cpu", "remote"],
        help=f"Compute backend (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI",
    )
    parser.add_argument(
        "--advance_frames",
        type=int,
        default=30,
        help="Number of simulation frames to advance in headless mode (default: 30)",
    )
    return parser


# ---------------------------------------------------------------------------
# Batch test runner (executed only when run as __main__)
# ---------------------------------------------------------------------------

def _build_batch_parser() -> argparse.ArgumentParser:
    """Parser specific to the batch runner (adds --frames / --verbose)."""
    parser = create_default_parser()
    parser.set_defaults(advance_frames=120)
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of simulation frames per test (default: 120)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print test output to stdout in addition to log files",
    )
    return parser


def _run_batch() -> int:
    parser = _build_batch_parser()
    args = parser.parse_args()

    tests_dir = Path(PROJECT_ROOT) / "PythonBindings" / "tests"
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(tests_dir))

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, bool, int, float]] = []
    start_all = time.time()

    for test_file in TESTS:
        test_name = test_file.replace(".py", "")
        test_path = tests_dir / test_file
        out_dir = OUTPUT_ROOT / test_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"

        title = f"  {test_name}  "
        line = "=" * max(60, len(title) + 4)
        print(f"\n{line}")
        print(f"  {test_name}")
        print(f"{line}")

        cmd = [
            sys.executable,
            str(test_path),
            "--headless",
            "--advance_frames", str(args.frames),
            "--backend", args.backend,
        ]
        print(f"  $ {' '.join(cmd)}")
        t0 = time.time()

        try:
            with open(log_path, "w") as log:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=600,
                    cwd=str(tests_dir),
                )
            elapsed = time.time() - t0
            ok = proc.returncode == 0

            if args.verbose or not ok:
                print(log_path.read_text()[-4000:] if not args.verbose else "")

            if ok:
                print(f"  [PASS]  ({elapsed:.1f}s)")
            else:
                print(f"  [FAIL]  exit={proc.returncode}  ({elapsed:.1f}s)")

            results.append((test_name, ok, proc.returncode, elapsed))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  [TIMEOUT]  ({elapsed:.1f}s)")
            results.append((test_name, False, -1, elapsed))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  [ERROR]  {exc}  ({elapsed:.1f}s)")
            results.append((test_name, False, -1, elapsed))

    total_elapsed = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = len(results) - passed
    for name, ok, rc, t in results:
        status = "PASS" if ok else f"FAIL({rc})"
        print(f"  [{status}]  {name}  ({t:.1f}s)")
    print(f"\n  {passed}/{len(results)} passed, {failed} failed  [{total_elapsed:.1f}s total]")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_batch())

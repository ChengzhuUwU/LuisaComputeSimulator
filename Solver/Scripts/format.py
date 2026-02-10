import os
import subprocess
import threading

# Top-level directories (relative to the repository root) to format
TARGET_DIRS = [
    "UnitTest",
    "Solver",
    "Application",
    "PythonBindings",
]

# Allow overriding clang-format via env var
clang_format_path = os.environ.get("CLANG_FORMAT", "clang-format")

file_types = [".cpp", ".h", ".hpp", ".c", ".cc", ".cxx", ".hxx"]


def run_clang_format(file_list):
    for file_path in file_list:
        print(f"Formatting file: {file_path}")
        try:
            subprocess.check_call([clang_format_path, "-i", file_path, "--style=file"])
            print(f"Formatting completed: {file_path}")
        except Exception as e:
            print(f"Formatting failed for {file_path}: {e}")


def traverse_and_collect(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(tuple(file_types)):
                file_list.append(file_path)
                if len(file_list) >= 10:
                    # copy list for thread to avoid mutation races
                    batch = file_list.copy()
                    thread = threading.Thread(target=run_clang_format, args=(batch,))
                    thread.start()
                    file_list = []
    if file_list:
        thread = threading.Thread(target=run_clang_format, args=(file_list.copy(),))
        thread.start()


def main():
    # repo root is two levels up from this script (Solver/Scripts)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    threads = []
    for d in TARGET_DIRS:
        dir_path = os.path.join(repo_root, d)
        if os.path.isdir(dir_path):
            print(f"Scanning directory: {dir_path}")
            t = threading.Thread(target=traverse_and_collect, args=(dir_path,))
            t.start()
            threads.append(t)
        else:
            print(f"Directory not found, skipping: {dir_path}")

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
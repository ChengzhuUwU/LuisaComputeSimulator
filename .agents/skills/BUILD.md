## Basic Building Operation

If you are coding agent, before configuration, you need to read the configuration params in [lcs_config.ini](/lcs_config.ini) (If the file not exists, you need to create one from copying [lcs_config_template.ini](/lcs_config_template.ini)). If value of the key is not empty, you need to apply the value in configuration:
- You can ignore the param if value is empty: `CMAKE_BUILD_TYPE = `
- You need to apply the param if value is not empty: `LCS_PYTHON_EXECUTABLE = /opt/homebrew/bin/python3`

Build the project:

```bash
cmake -S . -B build 
cmake --build build -j
```

For python test, most of the time you need to add the launching param `--headless`. For example:

```bash
<LCS_PYTHON_EXECUTABLE> PythonBindings/tests/test_cloth_soft_rigid_coupling.py --headless
```

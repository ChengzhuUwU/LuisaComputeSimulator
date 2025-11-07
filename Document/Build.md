# Build

## For linux users

If the cmake version is older than 3.26 (Check it using ``cmake --version``), you need to install a recent prebuilt release from https://cmake.org/files/LatestRelease:

```bash
sudo apt-get remove --purge -y cmake
# You may need to replace the url for correct version
wget https://cmake.org/files/LatestRelease/cmake-4.1.0-linux-x86_64.tar.gz
tar -xf cmake-4.1.0-linux-x86_64.tar.gz
# symlink the cmake binary into your PATH
sudo ln -s "$(pwd)/cmake-4.1.0-linux-x86_64/bin/cmake" /usr/local/bin/cmake
cmake --version
```

## Support build matrix

|          | windows | ubantu | macos|
|  -----   |---------|--------|------|
| gcc-11   |  [ ]    |   [ ]  | [ ]  |
| gcc-15   |  [ ]    |   [ ]  | [ ]  |
| clang-14 |  [ ]    |   [ ]  | [ ]  |
| clang-18 |  [ ]    |   [ ]  | [ ]  |
| msvc(cl) |  [ ]    |        |      |
<!-- | clang-cl |  [ ]    |   [ ]  | [ ]  | -->


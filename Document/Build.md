# Build



## For Linux Users

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



<!-- | clang-cl |  [ ]    |   [ ]  | [ ]  | -->


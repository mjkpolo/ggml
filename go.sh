#!/bin/bash

setup() {
  HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -G Ninja -B build \
  -DGGML_HIPBLAS=ON \
  -DAMDGPU_TARGETS=gfx908 \
  -DCMAKE_BUILD_TYPE=Release
}

build() {
  cmake --build build --config Release --target demo
}

run() {
  ./build/bin/demo
}

if [[ -z $1 ]]
then
  build
  run
else
  $@
fi

#!/bin/sh
set -eu
cd "$(dirname "$0")"
prefix=$(cd ../../../install && pwd)
deps="${HEXAMER_DEPS:?Set HEXAMER_DEPS to the environment prefix containing libmetatensor}"
/usr/bin/clang++ -std=c++17 -shared -fPIC hexamer.cpp -o hexamer.dylib \
    -I"$prefix/include" -I"$prefix/include/metatomic/third-party" -I"$deps/include" \
    -L"$prefix/lib" -lmetatomic -L"$deps/lib" -lmetatensor \
    -Wl,-rpath,"$prefix/lib" -Wl,-rpath,"$deps/lib"

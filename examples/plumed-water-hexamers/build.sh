#!/bin/sh
set -eu
cd "$(dirname "$0")"
example_dir=$(pwd)
repo_dir=$(cd ../.. && pwd)
work_dir=$(cd ../../.. && pwd)
prefix="$work_dir/install"
deps="${HEXAMER_DEPS:?Set HEXAMER_DEPS to the environment prefix containing libmetatensor}"
cargo_exe="${HEXAMER_CARGO:-$HOME/.cargo/bin/cargo}"
export PATH="$(dirname "$cargo_exe"):$PATH"

# Keep the native version pinned even when this documentation branch advances.
core_dir="$work_dir/metatomic-pinned"
if test ! -d "$core_dir"; then
    git -C "$repo_dir" worktree add --detach "$core_dir" b02b9ff3df6458cca21fa8690d229bbda97e6f78
fi
test "$(git -C "$core_dir" rev-parse HEAD)" = b02b9ff3df6458cca21fa8690d229bbda97e6f78
git -C "$core_dir" diff --exit-code HEAD -- metatomic-core python/metatomic_core
test "$(git -C "$work_dir/plumed2" rev-parse HEAD)" = 8226d8f682372750f46b3ed859c37e3dc0657803
# Keep the yanked zip 8.6.0 dependency pinned without changing core source.
if test ! -f "$core_dir/Cargo.lock"; then
    cp Cargo.lock.pinned "$core_dir/Cargo.lock"
fi
cmake -S "$core_dir/metatomic-core" -B "$core_dir/build-core" \
    -DCARGO_EXE="$cargo_exe" -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_PREFIX_PATH="$deps"
cmake --build "$core_dir/build-core" -j 6
cmake --install "$core_dir/build-core"

cd "$work_dir/plumed2"
if git apply --check "$example_dir/plumed-core-api.patch" 2>/dev/null; then
    git apply "$example_dir/plumed-core-api.patch"
else
    git apply --reverse --check "$example_dir/plumed-core-api.patch"
fi
./configure --enable-libmetatomic-core --disable-libtorch --disable-libmetatomic \
    --disable-mpi --enable-modules=none:+metatomic:+colvar:+xdrfile:+molfile:+cltools:+generic:+setup:+bias --prefix="$prefix" \
    CXX=/usr/bin/clang++ CC=/usr/bin/clang \
    CPPFLAGS="-I$prefix/include -I$prefix/include/metatomic/third-party -I$deps/include" \
    LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib -L$deps/lib -Wl,-rpath,$deps/lib"
make -j 6
cd "$example_dir"
/usr/bin/clang++ -std=c++17 -shared -fPIC hexamer.cpp -o hexamer.dylib \
    -I"$prefix/include" -I"$prefix/include/metatomic/third-party" -I"$deps/include" \
    -L"$prefix/lib" -lmetatomic -L"$deps/lib" -lmetatensor \
    -Wl,-rpath,"$prefix/lib" -Wl,-rpath,"$deps/lib"

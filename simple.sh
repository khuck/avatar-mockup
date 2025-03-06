set -e
# Build:

export Kokkos_ROOT=$HOME/src/albany/trilinos-install
#export Kokkos_ROOT=$HOME/src/apex-kokkos-tuning/install
#export Kokkos_ROOT=/Users/khuck/spack/opt/spack/darwin-sequoia-m1/apple-clang-16.0.0/kokkos-4.5.00-4xk5w5d7dkayb2hcf27xq5kbep77wom7
cmake -DCMAKE_PREFIX_PATH=$Kokkos_ROOT -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# Run:

export OMP_PLACES=threads
export OMP_PROC_BIND=spread
if [ -f ./build/src/libsimple-tuner.so ] ; then
    export KOKKOS_TOOLS_LIBS=./build/src/libsimple-tuner.so
elif [ -f ./build/src/libsimple-tuner.dylib ] ; then
    export KOKKOS_TOOLS_LIBS=./build/src/libsimple-tuner.dylib
fi

#./build/src/meta-smoother
./build/src/meta-smoother-discrete


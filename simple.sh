# Build:

export Kokkos_ROOT=$HOME/src/albany/apex-kokkos-tuning/install
cmake -DCMAKE_PREFIX_PATH=$Kokkos_ROOT -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# Run:

export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export KOKKOS_TOOLS_LIBS=./build/src/libsimple-tuner.so

./build/src/meta-smoother


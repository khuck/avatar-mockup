# avatar-mockup
simple implementation of a dummy meta-smoother and simple tuner that uses random values.

To run the example, edit simple.sh to change the location of the Kokkos installation directory, and then run the simple.sh script. The meta-smoother will run 300 times so that each smoother can be "run" 100 times each, and each time the simple tuner will choose random values for each tunable parameter.

Sample output:

```
Options for MTGS: Number of Sweeps [1,2]
Options for MTGS: Damping Factor[0.800000,1.200000]
Options for TSGS: Number of Sweeps [1,2]
Options for TSGS: Inner Damping Factor[0.800000,1.200000]
Options for C: Chebyshev Degree [1,2,3,4,5,6]
Options for C: Eigenvalue Ratio[10.000000,50.000000]
Options for C: Maximum Chebychev Iterations [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]

C: Chebyshev Degree target value: 5
C: Eigenvalue Ratio target value: 15
C: Maximum Chebychev Iterations target value: 75
MTGS: Number of Sweeps target value: 1
MTGS: Damping Factor target value: 0.9
TSGS: Number of Sweeps target value: 2
TSGS: Inner Damping Factor target value: 1.1

Best random value for variable meta-smoother: 2
Best random value for variable MTGS: Number of Sweeps: 1
Best random value for variable MTGS: Damping Factor: 0.962707
Best random value for variable TSGS: Number of Sweeps: 2
Best random value for variable TSGS: Inner Damping Factor: 0.949422
Best random value for variable C: Chebyshev Degree: 5
Best random value for variable C: Eigenvalue Ratio: 10.1665
Best random value for variable C: Maximum Chebychev Iterations: 85
```

To see lots and lots of output, set `export KOKKOS_VERBOSE=1` before running. to turn that back off, `unset KOKKOS_VERBOSE`.

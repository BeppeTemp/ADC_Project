# High Performance Computing's Project

## How to run

- In vscode use task (ctrl + b) to run C, C++ and CUDA Programs.
- On linux systems use **nvtop** to see realtime stat about GPU on your system.
- Profiler: nsyght system (to run use nsys profile app)

## Device info

| Device name                         | NVIDIA GeForce RTX 2060 |
| ----------------------------------- | ----------------------- |
| Compute capability                  | 7.5                     |
| Clock Rate                          | 1755000 kHz             |
| Total SMs                           | 30                      |
| Shared Memory Per SM                | 65536 bytes             |
| Registers Per SM                    | 65536 32-bit            |
| Max threads per SM                  | 1024                    |
| L2 Cache Size                       | 3145728 bytes           |
| Total Global Memory                 | 6214582272 bytes        |
| Memory Clock Rate                   | 7001000 kHz             |
| Max threads per block               | 1024                    |
| Max threads in X-dimension of block | 1024                    |
| Max threads in Y-dimension of block | 1024                    |
| Max threads in Z-dimension of block | 64                      |
| Max blocks in X-dimension of grid   | 2147483647              |
| Max blocks in Y-dimension of grid   | 65535                   |
| Max blocks in Z-dimension of grid   | 65535                   |
| Shared Memory Per Block             | 49152 bytes             |
| Registers Per Block                 | 65536 32-bit            |
| Warp size                           | 32                      |

numero di blocchi massimo 16
matrix[row][col] = array[row * size + col]

//Convolution reference
https://ulhpc-tutorials.readthedocs.io/en/latest/cuda/exercises/convolution/

g++ -o bench src/main.cpp src/mm_cpu.cpp src/conv_cpu.cpp -I.


-lboost_program...
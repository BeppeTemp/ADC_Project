# Tensor Unit Evaluation

The purpose of the project is present a simple benchmark suit, of **NVIDIA Tensor Core**. The benchmark suite is written in **C++** and use **Boost Programm Options** library for manage various program flag that allow to customize benchmark execution:

```txt
Allowed options:
  --help                      Produce help message.
  -c [ --exclude-cpu ]        Activate CPU tests.
  -g [ --exclude-gpu ]        Exclude GPU tests.
  -t [ --exclude-tensor ]     Exclude MMA tests.
  -i [ --iteration ] arg (=5) Set iteration number for avg calculation.
```

### Operation supported

The benchmark is based on two different operation: **Convolution** and **Matrix Multiplication**. More in details in the code there are 2 version (**CPU** and **GPU**) for the **Convolution** code and 3 versione (**CPU**, **GPU** and **Tensor**) for the **Matrix Multiplication**.

### Sizes

For what concern the sizes, the benchmark is executed on square matrix of 5 predefined sizes:
* **1024 * 1024**
* **2048 * 2048**
* **4098 * 4098**
* **8192 * 8192**
* **16384 * 16384**

In the case of convolution, **mask** is a square matrix **5 * 5** with **center** in position **[2,2]**.

## Output

At the end of computation, the result of benchmark is presented in the file **Report.txt**.

## Build and Run Instruction

In order to build the project clone the repository and run the following command:

```bash
make
```
After this for execute run the following command:

```bash
./Tensor_Bench <flags>
```

### Different GPU Architecture (Compute Capability)

The project is build and tested on the **Compute Capability 7.5** if you have different **CC** you need to change compilation flag in the **makefile** as follow:

```makefile
-gencode=arch=compute_<your CC version>,code=sm_<your cc version> 
```

<ins>**ATTENTION**</ins>: **CC** under **7.0** not support **Tensor Cores**.
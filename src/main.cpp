#include <cuda.h>
#include <mma.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#define PRINT_GREEN(str) printf("\x1b[32m%s\x1b[0m", str);
#define PRINT_RED(str) printf("\x1b[31m%s\x1b[0m", str);

#include "../include/conv_op.cuh"
#include "../include/matrix_op.cuh"

using namespace boost::program_options;
using namespace std;

// Report file object
class File_Report {
   public:
    ofstream report;
    File_Report() { report.open("Report.txt"); }
    void write(string str) { report << str; }
    ~File_Report() { report.close(); }
};

// Functions
void printMat(float* mat, int size) {
    // Print the entire matrix
    printf("\n");
    for (int i = 0; i < (size * size); i++) {
        printf("|");
        printf("%05.2f", mat[i]);
        if (((i + 1) % (size) == 0) && (i != 0))
            printf("|\n");
        if ((size * size) == 1)
            printf("|\n");
        if (size == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}
string time_stats(double seconds) {
    return string("Execution times:\n") + string("\t* ") + to_string(seconds * 1000 * 1000) + string(" μs\n") + string("\t* ") + to_string(seconds * 1000) + string(" ms\n") + string("\t* ") + to_string(seconds) + string(" s\n") + string("\n");
}

int main(int argc, char* argv[]) {
    int size;

    // Declare the supported options.
    options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")("size", value<int>()->required(), "set size of matrix");

    try {
        // Parse the command line arguments.
        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);

        // Print help message.
        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        size = vm["size"].as<int>();
        // Multiplo di 16 massimo testato 16384

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    File_Report fr;
    float* mat_res;

    // Convolution section
    float* mat_start = (float*)malloc(size * size * sizeof(float));
    float* mask = (float*)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));

    for (int i = 0; i < size * size; i++)
        mat_start[i] = 1;

    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++)
        mask[i] = 1;

    float* mat_res_cpu = (float*)calloc(size * size, sizeof(float));
    fr.write("Convolution using CPU\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(conv_cpu(mat_start, mask, mat_res_cpu, size)));
    printf("Convolution on CPU completed.\n");
    // printMat(mat_res_cpu, size);

    float* mat_res_gpu = (float*)calloc(size * size, sizeof(float));
    fr.write("Convolution using GPU\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(conv_gpu(mat_start, mask, mat_res_gpu, size)));
    printf("Convolution on GPU completed.\n");
    // printMat(mat_res_gpu, size);

    printf("Result (");
    if (conv_checker(mat_res_cpu, mat_res_gpu, size)) {
        PRINT_GREEN("Correct");
    } else {
        PRINT_RED("Incorrect");
    }
    printf(").\n");
    
    free(mat_start);
    free(mask);
    free(mat_res_cpu);
    free(mat_res_gpu);

    // Matrix Multiplication section
    float* mat_a = (float*)malloc(size * size * sizeof(float));
    float* mat_b = (float*)malloc(size * size * sizeof(float));

    for (int i = 0; i < size * size; i++) {
        mat_a[i] = 1;
        mat_b[i] = 1;
    }

    mat_res = (float*)calloc(size * size, sizeof(float));
    fr.write("Matrix multiplication using CPU\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(mm_cpu(mat_a, mat_b, mat_res, size)));

    printf("Matrix multiplication on CPU completed (");
    if (mm_checker(mat_res, size)) {
        PRINT_GREEN("Correct");
    } else {
        PRINT_RED("Incorrect");
    }
    printf(").\n");

    printMat(mat_res, size);
    free(mat_res);

    mat_res = (float*)calloc(size * size, sizeof(float));
    fr.write("Matrix multiplication using GPU\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(mm_gpu(mat_a, mat_b, mat_res, size)));

    printf("Matrix multiplication on GPU completed (");
    if (mm_checker(mat_res, size)) {
        PRINT_GREEN("Correct");
    } else {
        PRINT_RED("Incorrect");
    }
    printf(").\n");

    printMat(mat_res, size);
    free(mat_res);

    free(mat_a);
    free(mat_b);

    half* mat_a_half = (half*)malloc(size * size * sizeof(half));
    half* mat_b_half = (half*)malloc(size * size * sizeof(half));

    for (int i = 0; i < size * size; i++) {
        mat_a_half[i] = __float2half(1);
        mat_b_half[i] = __float2half(1);
    }

    mat_res = (float*)calloc(size * size, sizeof(float));
    fr.write("Matrix multiplication using Tensor\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(mm_tensor(mat_a_half, mat_b_half, mat_res, size)));

    printf("Matrix multiplication on Tensor completed (");
    if (mm_checker(mat_res, size)) {
        PRINT_GREEN("Correct");
    } else {
        PRINT_RED("Incorrect");
    }
    printf(").\n");

    printMat(mat_res, size);
    free(mat_res);

    free(mat_a_half);
    free(mat_b_half);

    return 0;
}
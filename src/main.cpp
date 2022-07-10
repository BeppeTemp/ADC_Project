#include <cuda.h>
#include <mma.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#define PRINT_GREEN(str) printf("\x1b[32m%s\x1b[0m", str);
#define PRINT_RED(str) printf("\x1b[31m%s\x1b[0m", str);

#define N_TEST 5

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
    // options_description desc("Allowed options");
    // desc.add_options()("help", "produce help message")("size", value<int>()->required(), "set size of matrix");

    // try {
    //     variables_map vm;
    //     store(parse_command_line(argc, argv, desc), vm);
    //     notify(vm);

    //     if (vm.count("help")) {
    //         cout << desc << "\n";
    //         return 1;
    //     }

    //     size = vm["size"].as<int>();
    // } catch (const std::exception& e) {
    //     std::cerr << e.what() << std::endl;
    //     return 1;
    // }

    File_Report fr;

    // int sizes[N_TEST] = {1024, 2048, 4096, 8192, 16384};
    int sizes[N_TEST] = {1024, 2048, 1024, 2048, 1024};

    for (int k = 0; k < N_TEST; k++) {
        printf("Starting test %d size %d\n", k + 1, sizes[k]);
        printf("----------------------------------------------------\n");

        fr.write("Starting test " + to_string(k + 1) + " size " + to_string(sizes[k]) + "\n");
        fr.write("----------------------------------------------------\n");

        float* mat_res;

        // -------------------------------
        // Convolution section
        // -------------------------------
        float* mat_start = (float*)malloc(sizes[k] * sizes[k] * sizeof(float));
        float* mask = (float*)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));

        double conv_avg_cpu = 0;
        double conv_avg_gpu = 0;

        #pragma omp parallel for
        for (int i = 0; i < sizes[k] * sizes[k]; i++)
            mat_start[i] = 1;

        for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++)
            mask[i] = 1;

        for (int i = 0; i < N_TEST; i++) {
            // Convolution on CPU
            float* mat_res_cpu = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
            conv_avg_cpu += conv_cpu(mat_start, mask, mat_res_cpu, sizes[k]);

            // Convolution on GPU
            float* mat_res_gpu = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
            conv_avg_gpu += conv_gpu(mat_start, mask, mat_res_gpu, sizes[k]);

            printf("Convolution CPU & GPU interation %d completed, result: ", i + 1);
            if (conv_checker(mat_res_cpu, mat_res_gpu, sizes[k])) {
                PRINT_GREEN("Correct");
            } else {
                PRINT_RED("Incorrect");
            }
            printf("\n");

            free(mat_res_cpu);
            free(mat_res_gpu);
        }
        printf("\n");

        conv_avg_cpu /= N_TEST;
        conv_avg_gpu /= N_TEST;

        fr.write("Convolution using CPU\n");
        fr.write(time_stats(conv_avg_cpu));
        fr.write("Convolution using GPU\n");
        fr.write(time_stats(conv_avg_gpu));

        free(mat_start);
        free(mask);

        // -------------------------------
        // Matrix Multiplication section
        // -------------------------------
        float* mat_a = (float*)malloc(sizes[k] * sizes[k] * sizeof(float));
        float* mat_b = (float*)malloc(sizes[k] * sizes[k] * sizeof(float));

        half* mat_a_half = (half*)malloc(sizes[k] * sizes[k] * sizeof(half));
        half* mat_b_half = (half*)malloc(sizes[k] * sizes[k] * sizeof(half));

        double mm_avg_cpu = 0;
        double mm_avg_gpu = 0;
        double mm_avg_tensor = 0;

        #pragma omp parallel for
        for (int i = 0; i < sizes[k] * sizes[k]; i++) {
            mat_a[i] = 1;
            mat_b[i] = 1;
            mat_a_half[i] = __float2half(1);
            mat_b_half[i] = __float2half(1);
        }

        for (int i = 0; i < N_TEST; i++) {
            // Matrix Multiplication on CPU
            mat_res = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
            mm_avg_cpu += mm_cpu(mat_a, mat_b, mat_res, sizes[k]);

            printf("Matrix multiplication CPU interation %d completed, result: ", i + 1);
            if (mm_checker(mat_res, sizes[k])) {
                PRINT_GREEN("Correct");
            } else {
                PRINT_RED("Incorrect");
            }
            printf("\n");

            free(mat_res);
        }
        printf("\n");

        for (int i = 0; i < N_TEST; i++) {
            // Matrix Multiplication on GPU
            mat_res = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
            mm_avg_gpu += mm_gpu(mat_a, mat_b, mat_res, sizes[k]);

            printf("Matrix multiplication GPU interation %d completed, result: ", i + 1);
            if (mm_checker(mat_res, sizes[k])) {
                PRINT_GREEN("Correct");
            } else {
                PRINT_RED("Incorrect");
            }
            printf("\n");

            free(mat_res);
        }
        printf("\n");

        for (int i = 0; i < N_TEST; i++) {
            // Matrix Multiplication on Tensor
            mat_res = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
            mm_avg_tensor += mm_tensor(mat_a_half, mat_b_half, mat_res, sizes[k]);

            printf("Matrix multiplication Tensor interation %d completed, result: ", i + 1);
            if (mm_checker(mat_res, sizes[k])) {
                PRINT_GREEN("Correct");
            } else {
                PRINT_RED("Incorrect");
            }
            printf("\n");

            free(mat_res);
        }

        mm_avg_cpu /= N_TEST;
        mm_avg_gpu /= N_TEST;
        mm_avg_tensor /= N_TEST;

        fr.write("Matrix Multiplication using CPU\n");
        fr.write(time_stats(mm_avg_cpu));
        fr.write("Matrix Multiplication using GPU\n");
        fr.write(time_stats(mm_avg_gpu));
        fr.write("Matrix Multiplication using Tensor\n");
        fr.write(time_stats(mm_avg_tensor));

        free(mat_a);
        free(mat_b);
        free(mat_a_half);
        free(mat_b_half);

        fr.write("\n");

        printf("\n");
    }

    return 0;
}
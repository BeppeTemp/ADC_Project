#include <cuda.h>
#include <mma.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#include "../include/matrix_op.cuh"

using namespace boost::program_options;
using namespace std;

class File_Report {
   public:
    ofstream report;
    File_Report() { report.open("Report.txt"); }
    void write(string str) { report << str; }
    ~File_Report() { report.close(); }
};

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

    float* mat_a = (float*)malloc(size * size * sizeof(float));
    float* mat_b = (float*)malloc(size * size * sizeof(float));
    float* mat_res = (float*)calloc(size * size, sizeof(float));

    for (int i = 0; i < size * size; i++) {
        mat_a[i] = 1;
        mat_b[i] = 1;
    }

    fr.write("Matrix multiplication using CPU\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(mm_cpu(mat_a, mat_b, mat_res, size)));

    fr.write("Matrix multiplication using GPU\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(mm_gpu(mat_a, mat_b, mat_res, size)));

    half* mat_a_half = (half*)malloc(size * size * sizeof(half));
    half* mat_b_half = (half*)malloc(size * size * sizeof(half));
    for (int i = 0; i < size * size; i++) {
        mat_a_half[i] = __float2half(1);
        mat_b_half[i] = __float2half(1);
    }

    fr.write("Matrix multiplication using Tensor\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(mm_tensor(mat_a_half, mat_b_half, mat_res, size)));

    return 0;
}
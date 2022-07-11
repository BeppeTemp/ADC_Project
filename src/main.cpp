#include <cuda.h>
#include <mma.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#include "../include/conv_op.cuh"
#include "../include/matrix_op.cuh"

using namespace boost::program_options;
using namespace std;

//Report file object
class File_Report {
public:
    ofstream report;
    File_Report() { report.open("Report.txt"); }
    void write(string str) { report << str; }
    ~File_Report() { report.close(); }
};

//Function to convert and print times
string time_stats(double seconds) {
    return string("Execution times ⏱: \n") + string("\t🔹 ") + to_string(seconds * 1000 * 1000) + string(" μs\n") + string("\t🔹 ") +
        to_string(seconds * 1000) + string(" ms\n") + string("\t🔹 ") + to_string(seconds) + string(" s\n") + string("\n");
}

int main(int argc, char* argv[]) {
    bool cpu_flag;
    bool gpu_flag;
    bool tensor_flag;

    int n_test;

    options_description desc("Allowed options");
    desc.add_options()
        ("help", "Produce help message.")
        ("exclude-cpu,c", bool_switch()->default_value(false), "Exclude CPU tests.")
        ("exclude-gpu,g", bool_switch()->default_value(false), "Exclude GPU tests.")
        ("exclude-tensor,t", bool_switch()->default_value(false), "Exclude MMA tests.")
        ("iteration,i", value<int>()->default_value(5), "Set iteration number for avg calculation.");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    try {
        notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        cpu_flag = vm["exclude-cpu"].as<bool>();
        gpu_flag = vm["exclude-gpu"].as<bool>();
        tensor_flag = vm["exclude-tensor"].as<bool>();

        n_test = vm["iteration"].as<int>();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    File_Report fr;

    int sizes[5] = { 1024, 2048, 4096, 8192, 16384 };
    // int sizes[5] = {1024, 1024, 1024, 1024, 1024};
    // int sizes[5] = { 32, 32, 32, 32, 32 };

    for (int k = 0; k < 5; k++) {
        printf("-------------------------------------------------------------\n");
        printf("\t\tℹ️  Starting test %d size %d ℹ️\n", k + 1, sizes[k]);
        printf("-------------------------------------------------------------\n");

        fr.write("---------------------------------------\n");
        fr.write("\tℹ️ Result test " + to_string(k + 1) + " (size " + to_string(sizes[k]) + ") ℹ️\n");
        fr.write("---------------------------------------\n");

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

        #pragma omp parallel for
        for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++)
            mask[i] = 1;

        // Convolution on CPU
        if (!cpu_flag) {
            for (int i = 0; i < n_test; i++) {
                float* mat_res_cpu = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
                conv_avg_cpu += conv_cpu(mat_start, mask, mat_res_cpu, sizes[k]);

                printf("Convolution CPU interation %d ✅\n", i + 1);
                free(mat_res_cpu);
            }
            printf("\n");
        }

        // Convolution on GPU
        if (!gpu_flag) {
            for (int i = 0; i < n_test; i++) {
                float* mat_res_gpu = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
                conv_avg_gpu += conv_gpu(mat_start, mask, mat_res_gpu, sizes[k]);

                printf("Convolution GPU interation %d ✅\n", i + 1);
                free(mat_res_gpu);
            }
            printf("\n");
        }

        conv_avg_cpu /= n_test;
        conv_avg_gpu /= n_test;

        if (!cpu_flag) {
            fr.write("🔸 Convolution using CPU\n");
            fr.write(time_stats(conv_avg_cpu));
        }
        if (!gpu_flag) {
            fr.write("🔸 Convolution using GPU\n");
            fr.write(time_stats(conv_avg_gpu / 1000));
        }

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

        // Matrix Multiplication on CPU
        if (!cpu_flag) {
            for (int i = 0; i < n_test; i++) {
                mat_res = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
                mm_avg_cpu += mm_cpu(mat_a, mat_b, mat_res, sizes[k]);

                printf("Matrix Multiplication CPU interation %d ", i + 1);
                mm_checker(mat_res, sizes[k]) ? printf("✅\n") : printf("❌\n");

                free(mat_res);
            }
            printf("\n");
        }

        // Matrix Multiplication on GPU
        if (!gpu_flag) {
            for (int i = 0; i < n_test; i++) {
                mat_res = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
                mm_avg_gpu += mm_gpu(mat_a, mat_b, mat_res, sizes[k]);

                printf("Matrix multiplication GPU interation %d ", i + 1);
                mm_checker(mat_res, sizes[k]) ? printf("✅\n") : printf("❌\n");

                free(mat_res);
            }
            printf("\n");
        }

        // Matrix Multiplication on Tensor
        if (!tensor_flag) {
            for (int i = 0; i < n_test; i++) {
                mat_res = (float*)calloc(sizes[k] * sizes[k], sizeof(float));
                mm_avg_tensor += mm_tensor(mat_a_half, mat_b_half, mat_res, sizes[k]);

                printf("Matrix multiplication Tensor interation %d ", i + 1);
                mm_checker(mat_res, sizes[k]) ? printf("✅\n") : printf("❌\n");

                free(mat_res);
            }
        }

        mm_avg_cpu /= n_test;
        mm_avg_gpu /= n_test;
        mm_avg_tensor /= n_test;

        if (!cpu_flag) {
            fr.write("Matrix Multiplication using CPU\n");
            fr.write(time_stats(mm_avg_cpu));
        }
        if (!gpu_flag) {
            fr.write("Matrix Multiplication using GPU\n");
            fr.write(time_stats(mm_avg_gpu / 1000));
        }
        if (!tensor_flag) {
            fr.write("Matrix Multiplication using Tensor\n");
            fr.write(time_stats(mm_avg_tensor / 1000));
        }

        free(mat_a);
        free(mat_b);
        free(mat_a_half);
        free(mat_b_half);

        fr.write("\n");
        printf("\n");
    }
    fr.write("Benchmark completed 😎 🎉\n");
    printf("Benchmark completed 😎 🎉\n");

    return 0;
}
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>

#include "../include/matrix_op.cuh"

using namespace boost::program_options;
using namespace std;

class File_Report {
   public:
    ofstream report;
    string myString;
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

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    File_Report fr;
 
    float* mat_res = (float*)calloc(size * size, sizeof(float));

    fr.write("Matrix multiplication using CPU\n");
    fr.write("Matrix Size: " + to_string(size) + "\n");
    fr.write(time_stats(mm_cpu(mat_res, size)));

    // cout << "Risultato matrice a * b: ";
    // printMat(mat_res, size);

    // int mask_size;

    // cout << "Inserire grandezza maschera: ";
    // cin >> mask_size;

    // float *mask, *mat_start;

    // mat_start = (float*)malloc(size * size * sizeof(float));
    // mask = (float*)malloc(mask_size * mask_size * sizeof(float));

    // conv_cpu(mat_start, mask, mat_res, size, mask_size, mask_size / 2);
    // cout << "Risultato conv: ";
    // printMat(mat_res, size);

    return 0;
}
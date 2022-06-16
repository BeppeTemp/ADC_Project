#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#define MASK_SIZE 4
#define MASK_CENTER 1

using namespace std;
using namespace std::chrono;

void time_stats(float micro_seconds)
{
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds);
    printf("    * %.2f ms \n", micro_seconds / 1000);
    printf("    * %.2f s \n", micro_seconds / 1000 / 1000);
    printf("\n");
}

void printMat(float *mat, int size)
{
    // Print the entire matrix
    printf("\n");
    for (int i = 0; i < (size * size); i++)
    {
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

int main()
{
    int sizes[1] = {4};

    float *mat_start, *mat_res;
    float *mask = (float *)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));
    float *mask_transp = (float *)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));

    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++)
    {
        if (i > 12)
        {
            mask[i] = 3;
        }
        else
        {
            mask[i] = 1;
        }
    }

    /*printf("Matrice iniziale\n");
    printMat(mask, MASK_SIZE);*/

    for (int i = 0; i < MASK_SIZE; i++)
    {
        for (int j = 0; j < MASK_SIZE; j++)
        {
            mask_transp[j * MASK_SIZE + i] = mask[i * MASK_SIZE + j];
        }
    }
    printf("Matrice trasposta\n");
    printMat(mask_transp, MASK_SIZE);

    for (int k = 0; k < 1; k++)
    {
        mat_start = (float *)malloc(sizes[k] * sizes[k] * sizeof(float));
        // mat_res = (float *)calloc(MASK_SIZE * MASK_SIZE, sizeof(float));
        mat_res = (float *)calloc(sizes[k] * sizes[k], sizeof(float));
        float *mat_temp = (float *)calloc(MASK_SIZE * MASK_SIZE, sizeof(float));

        for (int i = 0; i < sizes[k] * sizes[k]; i++)
        {
            mat_start[i] = 1;
        }
        printf("Matrice iniziale\n");
        printMat(mat_start, sizes[k]);

        #pragma omp parallel for
        for (int mat_row = 0; mat_row < sizes[k]; mat_row++)
        #pragma omp parallel for
            for (int mat_col = 0; mat_col < sizes[k]; mat_col++)
                for (int k_row = 0; k_row < MASK_SIZE; k_row++)
                    for (int k_col = 0; k_col < MASK_SIZE; k_col++)
                    {
                        int rel_row = mat_row + (k_row - MASK_CENTER);
                        int rel_col = mat_col + (k_col - MASK_CENTER);

                        if (rel_row >= 0 && rel_row < sizes[k] && rel_col >= 0 && rel_col < sizes[k])
                        {
                            //printf("rel row e col %d %d\n",rel_row,rel_col);
                            //mat_res[(mat_row * sizes[k]) + mat_col] += mat_start[(rel_row * sizes[k]) + rel_col] * mask_transp[(k_row * MASK_SIZE) + k_col];
                            mat_temp[(rel_row * MASK_SIZE) + rel_col] = mat_start[(rel_row * sizes[k]) + rel_col] * mask_transp[(k_row * MASK_SIZE) + k_col];
                            
                        //printf("Elemento per l'indice %d e %f\n",rel_row*MASK_SIZE+rel_col,mat_start[(rel_row * sizes[k]) + rel_col] * mask_transp[(k_row * MASK_SIZE) + k_col]);
                        //printMat(mat_temp,MASK_SIZE);
                        }
                        int sumTemp=0;
                        for (int i = 0; i < MASK_SIZE; i++)
                        {
                            for (int j = 0; j < MASK_SIZE; j++)
                            {
                                if(i==j)
                                {
                                    sumTemp+=mat_temp[(i*MASK_SIZE+j)];
                                    //printf("Valore per l'indice %d e %d e sum=%d\n",mat_row,mat_col,sum);
                                }
                            }
                        }
                        mat_res[mat_row*MASK_SIZE+mat_col]=sumTemp;
                    }

        printf("Matrice finale\n");
        printMat(mat_res, MASK_SIZE);

        /*for (int i = 0; i < sizes[k] * sizes[k]; i++)
        {
            mat_start[i] = 2;
        }
        for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++)
        {
            mat_res[i] = 0;
        }


        for (int i = 0; i < sizes[k]; i++)
        {
            for (int j = 0; j < MASK_SIZE; j++)
            {

                for (int h = 0; h < MASK_SIZE; h++)
                {
                    mat_res[i*MASK_SIZE+j] += mat_start[i*sizes[k]+h] * mask[j*MASK_SIZE+h];
                }
            }
        }*/

        auto start = high_resolution_clock::now();

        auto stop = high_resolution_clock::now();

        printf("Matrix size: %d x %d \n", sizes[k], sizes[k]);
        time_stats(duration_cast<microseconds>(stop - start).count());

        free(mat_start);
        free(mat_res);
    }
    free(mask);
    return 0;
}

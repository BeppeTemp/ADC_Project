#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

void printMat(float *mat, int m, int n)
{
    // Print the entire matrix
    printf("\n");
    for (int i = 0; i < m * n; i++)
    {
        printf("|");
        printf("%03.1f", mat[i]);
        if (((i + 1) % (n) == 0) && (i != 0))
            printf("|\n");
        if ((m * n) == 1)
            printf("|\n");
        if (n == 1 && ((i == 0)))
            printf("|\n");
    }

    printf("\n");
}

int main()
{

    float *mask = (float *)calloc(16 * 16, sizeof(float));
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            if (i < 4 && j < 4)
            {
                mask[(i * 16) + j] = 1;
            }
            else
            {
                mask[(i * 16) + j] = 0;
            }
        }
    }
    printMat(mask, 16, 16);

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
        {
            if (i < 4 && j < 4)
            {
                mask[(j * 16) + i] = mask[(i * 16) + j];
                // mask[(i * 16) + j] = mask[(j * 16) + i];
            }
        }

    int x = 0;

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
        {

            mask[(j * 16) * (i + 1)] = mask[(i * 16) + j];
        }

    printMat(mask, 16, 16);
}
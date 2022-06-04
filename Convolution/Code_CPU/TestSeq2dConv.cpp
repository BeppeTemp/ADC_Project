#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <chrono>

#define MASK_WIDTH2 5
#define MASK_WIDTH1 5
#define WIDTH1 7
#define WIDTH2 7

// find center position of kernel (half of kernel size)

int main()
{

    int mat_a[7][7]={1,2,3,4,5,6,7,2,3,4,5,6,7,8,3,4,5,6,7,8,9,4,5,6,7,8,5,6,5,6,7,8,5,6,7,6,7,8,9,0,1,2,7,8,9,0,1,2,3};
    int mat_b[5][5] = {1,2,3,2,1,2,3,4,3,2,3,4,5,4,3,2,3,4,3,2,1,2,3,2,1};

    int mat_res[7][7] = {0};

    int kCenterX = MASK_WIDTH2 / 2;
    int kCenterY = MASK_WIDTH1 / 2;
    int sum=0;

    for (int i = 0; i < WIDTH1; ++i) // rows
    {
        for (int j = 0; j < WIDTH2; ++j) // columns
        {
            for (int m = 0; m < MASK_WIDTH1; ++m) // kernel rows
            {
                int mm = MASK_WIDTH1 - 1 - m; // row index

                for (int n = 0; n < MASK_WIDTH2; ++n) // kernel columns
                {
                    int nn = MASK_WIDTH2 - 1 - n; // column index

                    // index of input signal, used for checking boundary
                    int ii = i + (m - kCenterY);
                    int jj = j + (n - kCenterX);

                    // ignore input samples which are out of bound
                    if (ii>= 0 && ii < WIDTH1 && jj >= 0 && jj < WIDTH2){
                    printf("Elementi moltiplicati input %d, kernel %d con riga=%d e colonna=%d \n",mat_a[ii][jj],mat_b[mm][nn],mm,nn);
                        mat_res[i][j] +=  mat_b[mm][nn]*mat_a[ii][jj];
                    }
                    /*else{
                        mat_res[i][j]=0;
                    }*/
                }
            }
        }
    }

     for (int i = 0; i <7; i++)
    {
        for (int j = 0; j < 7; j++)
        {
            printf("Elemento in posizione: riga %d e colonna %d è =%d\n",i,j,mat_res[i][j]);
        }
        
    }
}
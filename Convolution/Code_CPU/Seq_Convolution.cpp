#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <chrono>

#define PRINT_GREEN(str) printf("\x1b[32m%s\x1b[0m", str);
#define PRINT_RED(str) printf("\x1b[31m%s\x1b[0m", str);

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

int main()
{

    long nBytesFilter = 3 * 3 * sizeof(float);
    long nBytesMatrix = 5 * 5 * sizeof(float);

    float mat_a[5][5] = {0.5, 0.5, 0.3, 0.2, 0.1, 0.3, 0.2, 0.8, 0.9, 1.0, 0.7, 0.7, 0.5, 1.0, 1.0, 0.6, 0.6, 0.4, 0.9, 1.0, 0.1, 0.4, 0.6, 0.7, 0.8};
    float mat_b[3][3] = {0.12, 0.14, 0.22, 0.91, 0.44, 0.31, 0.77, 0.51, 0.13};

    float mat_res[3][3] = {0};
    

    

    auto start = high_resolution_clock::now();
    int offsetCol=0;
    int offsetRow=0;
    float sum=0.0;
    #pragma omp parallel for
    for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			int x = i;
			int y = j;

			// Kernel rows and columns are k and l respectively
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					// Convolute here.
                    
					sum += mat_b[k][l] * mat_a[x][y];
					y++; // Move right.
				}
				x++; // Move down.
				y = j; // Restart column position
			}
            
			mat_res[i][j] = sum; // Add result to output matrix.
			sum = 0.0; // Needed before we move on to the next index.
		}
	}
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            printf("Elemento in posizione: riga %d e colonna %d è =%f\n",i,j,mat_res[i][j]);
        }
        
    }
    
        
    
    auto stop = high_resolution_clock::now();

    

    time_stats(duration_cast<microseconds>(stop - start).count());

    free(mat_a);
    free(mat_b);
    free(mat_res);
return 0;
}


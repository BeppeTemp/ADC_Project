#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32
#define BLOCK_DIM 16

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 64
#define N_TILES 64
#define K_TILES 64

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)


//__global__ void WMMAINT8()
using namespace nvcuda;

using namespace std::chrono;

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds);
    printf("    * %.2f ms \n", micro_seconds / 1000);
    printf("    * %.2f s \n", micro_seconds / 1000 / 1000);
    printf("\n");
}
__host__ void InitMatrix(half *A, half *B, float *C)
{
	for (int i = 0; i < M_TOTAL*K_TOTAL; i++)
		A[i] = __float2half(rand() % 1000 / 1000.0f);
	for (int i = 0; i < K_TOTAL*N_TOTAL; i++)
		B[i] = __float2half(rand() % 1000 / 1000.0f);
	for (int i = 0; i < M_TOTAL*N_TOTAL; i++)
		C[i] = rand() % 1000 / 1000.0f;
}



__global__ void WMMAF16TensorCore(half *A, half *B, float *C,int size)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x)/WARP_SIZE;
	int iy = (blockIdx.y * blockDim.y + threadIdx.y);
	
    //Both matrix col_major 
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;
	wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
	
	wmma::fill_fragment(ab_frag, 0.0f);

	// AB = A*B
	int a_col, a_row, b_col, b_row, c_col, c_row;
	a_row = ix * M;
	b_col = iy * N;  //b_col=iy*N
	for (int k=0; k<size; k+=K) {
		a_col = b_row = k;  //b_row=k

		if (a_row < size && a_col < size && b_row < size && b_col < size) {
			// Load the inputs
			wmma::load_matrix_sync(a_frag, A + a_col + a_row * size, size);
			wmma::load_matrix_sync(b_frag, B + b_col + b_col * size, size);
            

			// Perform the matrix multiplication
            //ab_frag now holds the result for this warp’s output tile based on the multiplication of A and B.
			wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
		}
	}

	// D = AB + C
	c_col = b_col;
	c_row = a_row;
	if (c_row < M_TOTAL && c_col < N_TOTAL) {
		wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, wmma::mem_row_major);
        
		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
		}

		// Store the output
		wmma::store_matrix_sync(C + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, wmma::mem_row_major);
	}
}


int main(void) {
    int sizes[5] = {1024, 2048, 4096, 8192, 16384};

    half *mat_a_host, *mat_b_host;
    float *mat_res_host_gpu;
    half *mat_a_dev, *mat_b_dev;
    float *mat_res_dev;
    dim3 gridDim, blockDim;

    for (int i = 0; i < 5; i++) {
        long nBytes = sizes[i] * sizes[i] * sizeof(float);

        mat_a_host = (half*)malloc(nBytes);
        mat_b_host = (half*)malloc(nBytes);
        mat_res_host_gpu = (float*)malloc(nBytes);

        cudaMalloc((void**)&mat_a_dev, nBytes);
        cudaMalloc((void**)&mat_b_dev, nBytes);
        cudaMalloc((void**)&mat_res_dev, nBytes);

        for (int j = 0; j < sizes[i] * sizes[i]; j++) {
            mat_a_host[j] = __float2half((rand() % 1)+1);
            mat_b_host[j] = __float2half((rand() % 1)+1);
           
        }
        
        //InitMatrix(mat_a_host,mat_b_host,mat_res_host_gpu);

        cudaMemcpy(mat_a_dev, mat_a_host, nBytes, cudaMemcpyDefault);
        cudaMemcpy(mat_b_dev, mat_b_host, nBytes, cudaMemcpyDefault);
        cudaMemset(mat_res_dev, 0, nBytes);

        blockDim.x = BLOCK_DIM;
        blockDim.y = BLOCK_DIM;
        gridDim.x = sizes[i] / blockDim.x + ((sizes[i] % blockDim.x) == 0 ? 0 : 1);
        gridDim.y = sizes[i] / blockDim.y + ((sizes[i] % blockDim.y) == 0 ? 0 : 1);

        auto start = high_resolution_clock::now();
        WMMAF16TensorCore<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, mat_res_dev,sizes[i]);
        cudaDeviceSynchronize();
        auto stop = high_resolution_clock::now();

        cudaMemcpy(mat_res_host_gpu, mat_res_dev, nBytes, cudaMemcpyDeviceToHost);

        long check = 0;
        for (int k = 0; k < sizes[i] * sizes[i]; k++) {
            
            check += (long)mat_res_host_gpu[i];
            
        }

        printf("Matrix size: %d x %d \n", sizes[i], sizes[i]);
        printf("Block size: %d x %d = %d\n", BLOCK_DIM, BLOCK_DIM, BLOCK_DIM * BLOCK_DIM);
       // printf("Check: %ld\n", __half2float(check));
        time_stats(duration_cast<microseconds>(stop - start).count());

        free(mat_a_host);
        free(mat_b_host);

        cudaFree(mat_a_dev);
        cudaFree(mat_b_dev);
        cudaFree(mat_res_dev);
    }

    return 0;
}
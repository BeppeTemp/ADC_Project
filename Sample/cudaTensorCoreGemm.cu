#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 2
#define N_TILES 2
#define K_TILES 2

#define M_GLOBAL (M * M_TILES)  // 4096
#define N_GLOBAL (N * N_TILES)  // 4096
#define K_GLOBAL (K * K_TILES)  // 4096

#define debug_x 32
#define debug_y 1

using namespace nvcuda;

void printMat(float* mat, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        printf("|");
        printf(" %02.0f ", mat[i]);
        if (((i + 1) % (n) == 0) && (i != 0))
            printf("|\n");
        if ((m * n) == 1)
            printf("|\n");
        if (n == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}

void printMat(half* mat, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        printf("|");
        printf(" %02.0f ", __half2float(mat[i]));
        if (((i + 1) % (n) == 0) && (i != 0))
            printf("|\n");
        if ((m * n) == 1)
            printf("|\n");
        if (n == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}

__host__ void init_host_matrices(half* a, half* b, float* c) {
    // int k = 0;
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i * K_GLOBAL + j] = __float2half(1);
            // a[i * K_GLOBAL + j] = __float2half(k);
            // k++;
        }
    }

    // k = 0;
    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i * K_GLOBAL + j] = __float2half(1);
            // b[i * K_GLOBAL + j] = __float2half(k);
        }
        // k++;
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] = 0;
    }
}

__global__ void simple_wmma_gemm(half* a, half* b, float* c, float* d, int m_ld, int n_ld, int k_ld, float alpha, float beta) {
    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;  // 4096 Profondità
    int ldb = k_ld;  // 4096 Profondità
    int ldc = n_ld;  // 4096 Colonne

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // 32
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
        printf("tile_row: %d\n", warpM);
        printf("tile_col: %d\n", warpN);
        printf("\n");
    }

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K) {
        int aCol = i;
        int aRow = warpM * WMMA_M;
        int bCol = warpN * N;
        int bRow = i;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
                printf("Matrice A [%d,%d] da %d a %d, Matrice B [%d,%d] da %d a %d\n", aRow, aCol, aCol + aRow * lda, aCol + aRow * lda + ((lda * 16) - 1), bRow, bCol, bRow + bCol * ldb, bRow + bCol * ldb + ((ldb * 16) - 1));
            }

            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
        printf("\n");
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}

int main(int argc, char** argv) {
    printf("Initializing...\n");

    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);
    printf("\n");

    half* A_h = NULL;
    half* B_h = NULL;
    float* C_h = NULL;
    float* D_h = NULL;

    A_h = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    C_h = (float*)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
    D_h = (float*)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    half* A = NULL;
    half* B = NULL;
    float* C = NULL;
    float* D = NULL;

    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&C), sizeof(float) * M_GLOBAL * N_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&D), sizeof(float) * M_GLOBAL * N_GLOBAL);

    init_host_matrices(A_h, B_h, C_h);

    // printf("Mat A:\n");
    // printMat(A_h, M_GLOBAL, N_GLOBAL);
    // printf("Mat B:\n");
    // printMat(B_h, M_GLOBAL, N_GLOBAL);
    // printf("Mat C:\n");
    // printMat(C_h, M_GLOBAL, N_GLOBAL);

    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);

    const float alpha = 1.1f;
    const float beta = 1.2f;

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Griglia: %d, %d\n", gridDim.x, gridDim.y);
    printf("Blocco: %d, %d\n", blockDim.x, blockDim.y);
    printf("\n");

    printf("Thread [%d,%d]\n", debug_x, debug_y);
    printf("\n");

    simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);

    cudaMemcpy(D_h, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);

    // printf("Mat Res:\n");
    // printMat(D_h, M_GLOBAL, N_GLOBAL);

    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(reinterpret_cast<void*>(A));
    cudaFree(reinterpret_cast<void*>(B));
    cudaFree(reinterpret_cast<void*>(C));
    cudaFree(reinterpret_cast<void*>(D));

    return 0;
}

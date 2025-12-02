// Compilar: nvcc saxpy_cublas_vs_manual.cu -lcublas
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N (1 << 20)   // 1M elementos

__global__ void saxpy_kernel(int n, float a,
                             const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    float *d_x, *d_y_manual, *d_y_cublas;
    cudaMalloc(&d_x,        N * sizeof(float));
    cudaMalloc(&d_y_manual, N * sizeof(float));
    cudaMalloc(&d_y_cublas, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_manual, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_cublas, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 3.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_manual, ms_cublas;

    // --- SAXPY manual con kernel propio ---
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    saxpy_kernel<<<gridSize, blockSize>>>(N, alpha, d_x, d_y_manual);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_manual, start, stop);

    // --- SAXPY con cuBLAS ---
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEventRecord(start);
    // y = alpha * x + y
    cublasSaxpy(handle, N,
                &alpha,
                d_x, 1,
                d_y_cublas, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_cublas, start, stop);

    cudaMemcpy(h_y, d_y_cublas, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Tiempo SAXPY manual : %f ms\n", ms_manual);
    printf("Tiempo SAXPY cuBLAS : %f ms\n", ms_cublas);
    printf("Speedup cuBLAS/manual: %fx\n", ms_manual / ms_cublas);

    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y_manual);
    cudaFree(d_y_cublas);
    free(h_x);
    free(h_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// Compilar: nvcc cufft_vs_manual.cu -lcufft
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define N 4096

// Representación simple de complejo en float
typedef struct { float x, y; } cfloat;

__device__ cfloat cadd(cfloat a, cfloat b) {
    cfloat r; r.x = a.x + b.x; r.y = a.y + b.y; return r;
}
__device__ cfloat cmul(cfloat a, cfloat b) {
    cfloat r; r.x = a.x*b.x - a.y*b.y;
               r.y = a.x*b.y + a.y*b.x; return r;
}

// DFT naive O(N^2): cada hilo calcula una frecuencia k
__global__ void dft_naive(const cfloat *in, cfloat *out, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float TWO_PI = 6.283185307179586f;
    cfloat sum = {0.0f, 0.0f};
    for (int n0 = 0; n0 < n; ++n0) {
        float angle = -TWO_PI * k * n0 / n;
        cfloat w = {cosf(angle), sinf(angle)};
        sum = cadd(sum, cmul(in[n0], w));
    }
    out[k] = sum;
}

int main() {
    // Host
    cfloat *h_in  = (cfloat*)malloc(N * sizeof(cfloat));
    cfloat *h_out = (cfloat*)malloc(N * sizeof(cfloat));

    for (int i = 0; i < N; ++i) {
        h_in[i].x = sinf(2.0f * 3.14159f * i / N); // señal simple
        h_in[i].y = 0.0f;
    }

    // Device
    cfloat *d_in, *d_out_manual;
    cufftComplex *d_out_cufft;
    cudaMalloc(&d_in,         N * sizeof(cfloat));
    cudaMalloc(&d_out_manual, N * sizeof(cfloat));
    cudaMalloc(&d_out_cufft,  N * sizeof(cufftComplex));

    cudaMemcpy(d_in, h_in, N * sizeof(cfloat), cudaMemcpyHostToDevice);

    // Eventos para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_manual, ms_cufft;

    // --- Versión manual (DFT) ---
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    dft_naive<<<gridSize, blockSize>>>(d_in, d_out_manual, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_manual, start, stop);

    // --- Versión cuFFT ---
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    cudaEventRecord(start);
    cufftExecC2C(plan,
                 (cufftComplex*)d_in,
                 d_out_cufft,
                 CUFFT_FORWARD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_cufft, start, stop);

    // Copiar y mostrar tiempos
    cudaMemcpy(h_out, d_out_manual, N * sizeof(cfloat),
               cudaMemcpyDeviceToHost);

    printf("N = %d\n", N);
    printf("Tiempo DFT manual: %f ms\n", ms_manual);
    printf("Tiempo cuFFT:      %f ms\n", ms_cufft);
    printf("Speedup cuFFT/manual: %fx\n", ms_manual / ms_cufft);

    // Limpieza
    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out_manual);
    cudaFree(d_out_cufft);
    free(h_in);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

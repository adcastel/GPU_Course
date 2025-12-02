// Compilar (ejemplo): nvcc conv_cudnn_vs_manual.cu -lcudnn
#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(status) \
    if (status != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(status)); \
        return -1; \
    }

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("cuDNN error: %s\n", cudnnGetErrorString(status)); \
        return -1; \
    }

// Convolución 2D naive: 1 batch, 1 canal, padding=1, stride=1, kernel 3x3
__global__ void conv2d_naive(const float* __restrict__ in,
                             const float* __restrict__ k,
                             float* __restrict__ out,
                             int H, int W) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; // fila
    int x = blockIdx.x * blockDim.x + threadIdx.x; // columna
    if (y >= H || x >= W) return;

    float sum = 0.0f;
    // kernel 3x3, padding 1
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int iy = y + ky;
            int ix = x + kx;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                float val = in[iy * W + ix];
                float w   = k[(ky + 1) * 3 + (kx + 1)];
                sum += val * w;
            }
        }
    }
    out[y * W + x] = sum;
}

int main() {
    const int N = 1, C = 1, H = 128, W = 128;
    const int K = 1, R = 3, S = 3; // 1 filtro, 3x3
    const int pad_h = 1, pad_w = 1;
    const int stride_h = 1, stride_w = 1;

    size_t in_bytes   = N * C * H * W * sizeof(float);
    size_t filt_bytes = K * C * R * S * sizeof(float);
    size_t out_bytes  = N * K * H * W * sizeof(float); // mismo tamaño por padding=1

    // Host
    float *h_in   = (float*)malloc(in_bytes);
    float *h_filt = (float*)malloc(filt_bytes);
    float *h_out_manual = (float*)malloc(out_bytes);
    float *h_out_cudnn  = (float*)malloc(out_bytes);

    for (int i = 0; i < N*C*H*W; ++i)
        h_in[i] = 1.0f;   // imagen sencilla
    for (int i = 0; i < K*C*R*S; ++i)
        h_filt[i] = 1.0f; // kernel sencillo

    // Device
    float *d_in, *d_filt, *d_out_manual, *d_out_cudnn;
    CHECK_CUDA(cudaMalloc(&d_in, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_filt, filt_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_manual, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_cudnn, out_bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filt, h_filt, filt_bytes, cudaMemcpyHostToDevice));

    // Eventos
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms_manual, ms_cudnn;

    // --- Convolución manual ---
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    CHECK_CUDA(cudaEventRecord(start));
    conv2d_naive<<<grid, block>>>(d_in, d_filt, d_out_manual, H, W);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_manual, start, stop));

    // --- Convolución con cuDNN ---
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filt_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C, H, W));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        K, C, R, S));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w,
        stride_h, stride_w,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

    // Elegir algoritmo y workspace
    cudnnConvolutionFwdAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
        handle,
        in_desc, filt_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        in_desc, filt_desc, conv_desc, out_desc,
        algo,
        &workspace_bytes));

    void *d_workspace = NULL;
    if (workspace_bytes > 0)
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha,
        in_desc, d_in,
        filt_desc, d_filt,
        conv_desc, algo,
        d_workspace, workspace_bytes,
        &beta,
        out_desc, d_out_cudnn));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_cudnn, start, stop));

    // Copiar resultados y mostrar tiempos
    CHECK_CUDA(cudaMemcpy(h_out_manual, d_out_manual, out_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_cudnn,  d_out_cudnn,  out_bytes, cudaMemcpyDeviceToHost));

    printf("Salida manual[0]: %f\n", h_out_manual[0]);
    printf("Salida cuDNN [0]: %f\n", h_out_cudnn[0]);
    printf("Tiempo conv manual: %f ms\n", ms_manual);
    printf("Tiempo conv cuDNN : %f ms\n", ms_cudnn);
    printf("Speedup cuDNN/manual: %fx\n", ms_manual / ms_cudnn);

    // Limpieza
    if (d_workspace) cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    cudaFree(d_in);
    cudaFree(d_filt);
    cudaFree(d_out_manual);
    cudaFree(d_out_cudnn);

    free(h_in);
    free(h_filt);
    free(h_out_manual);
    free(h_out_cudnn);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

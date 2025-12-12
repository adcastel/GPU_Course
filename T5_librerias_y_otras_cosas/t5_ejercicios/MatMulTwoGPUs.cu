
/*************************************
 * Multiplication with multiGPU      *
 *************************************/

#include <stdio.h>
#include <cublas_v2.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }
#define CUBLAS_SAFE_CALL( call ) {                                         \
 cublasStatus_t err = call;                                                \
 if( CUBLAS_STATUS_SUCCESS != err ) {                                      \
   fprintf(stderr,"CUBLAS: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                              \
 } }


#define	A(i,j)		A[ (i) + ((j)*(m)) ]
#define	B(i,j)		B[ (i) + ((j)*(p)) ]
#define	C(i,j)		C[ (i) + ((j)*(m)) ]
#define	D(i,j)		D[ (i) + ((j)*(m)) ]

int main( int argc, char *argv[] ) {
  int m, n, p;
  int i, j;

  /* Generating input data */
  if( argc<4 ) {
    printf("Usage: %s m n p \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&m);
  sscanf(argv[2],"%d",&n);
  sscanf(argv[3],"%d",&p);
  if( m%2 ) {
    printf("%s: n must be multiple of 2\n",argv[0]);
    exit(-1);
  }
  float *A;
  CUDA_SAFE_CALL( cudaHostAlloc ( &A, m*p*sizeof(float), cudaHostAllocDefault ) );
  float *B;
  CUDA_SAFE_CALL( cudaHostAlloc ( &B, p*n*sizeof(float), cudaHostAllocDefault ) );
  printf("%s: Generating random matrices of size %dx%d and %dx%d...\n",argv[0],m,p,p,n);
  for( i=0; i<m; i++ ) {
    for( j=0; j<p; j++ ) {
      A( i, j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    }
  }
  for( i=0; i<p; i++ ) {
    for( j=0; j<n; j++ ) {
      B( i, j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    }
  }

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  cublasHandle_t handle0, handle1;
#pragma omp parallel sections
{
  #pragma omp section
  {
  CUDA_SAFE_CALL( cudaSetDevice( 0 ) ); // Set GPU to 0
  CUBLAS_SAFE_CALL( cublasCreate(&handle0) );
  }
  #pragma omp section
  {
  CUDA_SAFE_CALL( cudaSetDevice( 1 ) ); // Set GPU to 1
  CUBLAS_SAFE_CALL( cublasCreate(&handle1) );
  }
}

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL( cudaEventCreate(&start) );
  CUDA_SAFE_CALL( cudaEventCreate(&stop) );

  printf("%s: C=A*B with one GPU...\n",argv[0]);

  float *C;
  CUDA_SAFE_CALL( cudaHostAlloc ( &C, m*n*sizeof(float), cudaHostAllocDefault ) );
  float *D;
  CUDA_SAFE_CALL( cudaHostAlloc ( &D, m*n*sizeof(float), cudaHostAllocDefault ) ); /* Allocate memory for the resulting m-by-n matrix D in CPU memory */
  float *d_A, *d_B, *d_C;
  CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_A, m*p*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_B, p*n*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_C, m*n*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_A, A, m*p*sizeof(float), cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_B, B, p*n*sizeof(float), cudaMemcpyHostToDevice ) );
  CUBLAS_SAFE_CALL( cublasSgemm( handle0, CUBLAS_OP_N, CUBLAS_OP_N, m, n, p, &alpha, d_A, m, d_B, p, &beta, d_C, m ) );
  CUDA_SAFE_CALL( cudaMemcpy( C, d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost ) );
  CUDA_SAFE_CALL( cudaFree(d_A) );
  CUDA_SAFE_CALL( cudaFree(d_B) );
  CUDA_SAFE_CALL( cudaFree(d_C) );
  CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  float msecGPU = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecGPU, start, stop) );

  int k = m / 2;

  printf("%s: C=A*B with two GPUs ...\n",argv[0]);
  CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event

#pragma omp parallel sections
{
  #pragma omp section
  {
  float *d_A, *d_B, *d_C;
  CUDA_SAFE_CALL( /* Establecer la GPU 0 con cudaSetDevice */ ); // Set GPU to 0
  CUDA_SAFE_CALL( /* Reservar memoria en GPU para d_A de tamaño k*p con cudaMalloc */ 
  CUDA_SAFE_CALL( /* Reservar memoria en GPU para d_B de tamaño p*n con cudaMalloc */ 
  CUDA_SAFE_CALL( /* Reservar memoria en GPU para d_C de tamaño k*n con cudaMalloc */ 
  CUBLAS_SAFE_CALL( /* Enviar toda la matriz B (CPU) a la matriz d_B en GPU de tamaño p*n con cublasSetMatrix */ );
  CUBLAS_SAFE_CALL( /* Enviar el bloque superior de la matriz A (CPU) de tamaño k*p a la matriz d_A en GPU con cublasSetMatrix */ );
  CUBLAS_SAFE_CALL( /* Realizar el producto de matrices d_C = d_A * d_B en GPU con cublasSgemm */ );
  CUBLAS_SAFE_CALL( /* Enviar la matriz d_C (GPU) al bloque superior de la matriz D (CPU) de tamaño k*p con cublasSetMatrix */ );
  CUDA_SAFE_CALL( /* Liberar el espacio reservado para la matriz d_A con cudaFree */ ); 
  CUDA_SAFE_CALL( /* Liberar el espacio reservado para la matriz d_B con cudaFree */ ); 
  CUDA_SAFE_CALL( /* Liberar el espacio reservado para la matriz d_C con cudaFree */ ); 
  }

  #pragma omp section
  {
  float *d_A, *d_B, *d_C;
  CUDA_SAFE_CALL( /* Establecer la GPU 1 con cudaSetDevice */ ); // Set GPU to 1
  /***********************************************************************************************
    Realizar los mismos pasos que para la GPU teniendo en cuenta:
    1. Antes de realizar el producto se envía el bloque inferior de la matriz A (CPU) a d_A (GPU).
    2. Después de realizar el producto se envía d_C (GPU) al bloque inferior de D (CPU).
   ***********************************************************************************************/
  }
}

  CUDA_SAFE_CALL( cudaSetDevice( 0 ) ); // Set GPU to 0
  CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  float msecTwoGPU = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecTwoGPU, start, stop) );

  /* Check for correctness */
  float error = 0.0f;
  for( j=0; j<n; j++ ) {
    for( i=0; i<m; i++ ) {
      float a = fabs( C( i, j ) - D( i, j ) );
      error = a > error ? a : error;
    }
  }
  printf("Error = %.3e\n",error);
  double flops = 2.0 * (double) m * (double) n;
  double gigaFlops = (flops * 1.0e-9f) / (msecGPU / 1000.0f);
  double gigaFlopsTwoGPU = (flops * 1.0e-9f) / (msecTwoGPU / 1000.0f);
  printf("1 GPU time = %.2f msec.\n",msecGPU);
  printf("2 GPU time = %.2f msec.\n",msecTwoGPU);
  printf("1 GPU Gflops = %.2f \n",gigaFlops);
  printf("2 GPU Gflops = %.2f \n",gigaFlopsTwoGPU);
  
  CUDA_SAFE_CALL( cudaFreeHost( A ) );
  CUDA_SAFE_CALL( cudaFreeHost( B ) );
  CUDA_SAFE_CALL( cudaFreeHost( C ) );
  CUDA_SAFE_CALL( cudaFreeHost( D ) );
  CUBLAS_SAFE_CALL( cublasDestroy(handle0) );
  CUDA_SAFE_CALL( cudaSetDevice( 1 ) ); // Set GPU to 1
  CUBLAS_SAFE_CALL( cublasCreate(&handle1) );
  
}


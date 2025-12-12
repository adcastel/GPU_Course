
/*************************************
 * Matrix-Vector product CUDA kernel *
 * with and without streams          *
 *************************************/

#include <stdio.h>
#include <cublas_v2.h>

extern "C" {
  void saxpy_(const int *nz, const float *a, const float *x, const int *indx, float *y, const int *indy );
}

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
   cublasDestroy(handle);                                                  \
   exit(err);                                                              \
 } }

#define	A(i,j)		A[ (i) + ((j)*(m)) ]
#define	B(i,j)		B[ (i) + ((j)*(p)) ]
#define	C(i,j)		C[ (i) + ((j)*(m)) ]
#define	D(i,j)		D[ (i) + ((j)*(m)) ]

int main( int argc, char *argv[] ) {
  int m, n, p;
  int i, j;
  int nstreams;

  /* Generating input data */
  if( argc<5 ) {
    printf("Usage: %s m n p nstreams\n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&m);
  sscanf(argv[2],"%d",&n);
  sscanf(argv[3],"%d",&p);
  sscanf(argv[4],"%d",&nstreams);
  if( m%nstreams ) {
    printf("%s: n must be multiple of nstreams\n",argv[0]);
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

  float *d_A, *d_B, *d_C;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_A, m*p*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_B, p*n*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_C, m*n*sizeof(float) ) );
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  cublasHandle_t handle;
  CUBLAS_SAFE_CALL( cublasCreate(&handle) );

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL( cudaEventCreate(&start) );
  CUDA_SAFE_CALL( cudaEventCreate(&stop) );

  printf("%s: C=A*B in GPU...\n",argv[0]);
  float *C;
  CUDA_SAFE_CALL( cudaHostAlloc ( &C, m*n*sizeof(float), cudaHostAllocDefault ) );
  CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event
  CUDA_SAFE_CALL( cudaMemcpy( d_A, A, m*p*sizeof(float), cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_B, B, p*n*sizeof(float), cudaMemcpyHostToDevice ) );
  CUBLAS_SAFE_CALL( cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, p, &alpha, d_A, m, d_B, p, &beta, d_C, m ) );
  CUDA_SAFE_CALL( cudaMemcpy( C, d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost ) );
  CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  float msecGPU = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecGPU, start, stop) );

  printf("%s: C=A*B in GPU with streams ...\n",argv[0]);
  float *d_D;
  float *D;
  CUDA_SAFE_CALL( /* Allocate memory for the resulting m-by-n matrix D in CPU memory */ ); 
  CUDA_SAFE_CALL( /* Allocate memory into the GPU for an m-by-n matrix d_D */ );
  cudaStream_t stream[nstreams]; /* Array of CUDA streams */
  int k = m/nstreams; /* k is the row-block size */
  CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event
  CUDA_SAFE_CALL( /* Copy the p-by-n CPU matrix B into matrix d_B on GPU */ );
  for (int i = 0; i < nstreams; i++) {
    CUDA_SAFE_CALL( /* Create stream[i]. */ );
    /* Perform the computation with the streams */
    CUBLAS_SAFE_CALL( /* Send a k-by-p block of the CPU matrix A starting in row i*k to the corresponding position in the GPU matrix d_A. Use cublasSetMatrixAsync for simplicity. */ );
    CUBLAS_SAFE_CALL( /* Set stream stream[i] as current. */ );
    CUBLAS_SAFE_CALL( /* Perform the matrix multiplication of the just sent block in d_A by matrix d_B (cublasSgemm). Store the result into the corresponding block of d_D. */ );
    CUBLAS_SAFE_CALL( /* Send the resulting matrix of the last multiplication in d_D to the corresponding block into the CPU matrix D. Use cublasGetMatrixAsync for simplicity. */ );
    CUDA_SAFE_CALL( /* Destroy stream[i]. */ );
  }
  CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  float msecStreams = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecStreams, start, stop) );

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
  double gigaFlopsStreams = (flops * 1.0e-9f) / (msecStreams / 1000.0f);
  printf("GPU time without streams = %.2f msec.\n",msecGPU);
  printf("GPU time with streams = %.2f msec.\n",msecStreams);
  printf("Gflops without streams = %.2f \n",gigaFlops);
  printf("Gflops with streams = %.2f \n",gigaFlopsStreams);
  
  CUDA_SAFE_CALL( cudaFreeHost( A ) );
  CUDA_SAFE_CALL( cudaFreeHost( B ) );
  CUDA_SAFE_CALL( cudaFreeHost( C ) );
  CUDA_SAFE_CALL( cudaFreeHost( D ) );
  CUDA_SAFE_CALL( cudaFree(d_A) );
  CUDA_SAFE_CALL( cudaFree(d_B) );
  CUDA_SAFE_CALL( cudaFree(d_C) );
  CUDA_SAFE_CALL( cudaFree(d_D) );
  CUBLAS_SAFE_CALL( cublasDestroy(handle) );
  
}


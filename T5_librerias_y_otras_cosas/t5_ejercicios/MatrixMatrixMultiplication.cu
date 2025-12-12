
/*************************************
 * Matrix-Matrix product with CUBLAS *
 *************************************/

#include <stdio.h>
#include <...> /* Write here the name of the CUBLAS header file */

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

/* Matrices stored by columns: BLAS style */
#define	A(i,j)		A[ (i) + ((j)*(n)) ]
#define	B(i,j)		B[ (i) + ((j)*(n)) ]
#define	C(i,j)		C[ (i) + ((j)*(n)) ]
#define	gpu_C(i,j)	gpu_C[ (i) + ((j)*(n)) ]
#define	d_A(i,j) 	d_A[ (j) + ((i)*(n)) ]

int main( int argc, char *argv[] ) {
  int n;
  unsigned int i, j;

  /* Generating input data */
  if( argc<2 ) {
    printf("Usage: %s n \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);
  double *A = (double *) malloc( n*n*sizeof(double) );
  double *B = (double *) malloc( n*n*sizeof(double) );
  double *C = (double *) malloc( n*n*sizeof(double) );
  double *gpu_C = (double *) malloc( n*n*sizeof(double) );
  printf("%s: Generating two random matrices of size %dx%d...\n",argv[0],n,n);
  for( i=0; i<n; i++ ) {
    for( j=0; j<n; j++ ) {
      A( i, j ) = 2.0 * ( (double) rand() / RAND_MAX ) - 1.0;
    }
  }
  for( i=0; i<n; i++ ) {
    for( j=0; j<n; j++ ) {
      B( i, j ) = 2.0 * ( (double) rand() / RAND_MAX ) - 1.0;
    }
  }

  /* STARTUP CUBLAS context */
  /* Declare a CUBLAS handle with name handle */
  . . . 
  CUBLAS_SAFE_CALL( /* Create cublas handle */ );

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL( cudaEventCreate(&start) );
  CUDA_SAFE_CALL( cudaEventCreate(&stop) );

  const double ONE = 1.0;
  const double ZERO = 0.0;

  printf("%s: C=A*B in GPU...\n",argv[0]);
  // Allocate device memory
  double *d_A, *d_B, *d_C;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_A, n*n*sizeof(double) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_B, n*n*sizeof(double) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_C, n*n*sizeof(double) ) );
  CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event
  /* In this place transfer matrices A and B from Host to matrices d_A and d_B, respectively, on Device */
  CUDA_SAFE_CALL( /* A -> d_A */ );
  CUDA_SAFE_CALL( /* B -> d_B */ );
  CUBLAS_SAFE_CALL( /* In this place write a call to routine cublasDgemm that performs the matrix-matrix product on GPU */ );
  /* In this place transfer matrix d_C from Device to matrix gpu_C on the Host */
  CUDA_SAFE_CALL( /* d_C -> gpu_C */ );
  CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  float msecGPU = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecGPU, start, stop) );

  double flops = 2.0 * (double) n * (double) n * (double) n;
  float gigaFlopsGPU = (flops * 1.0e-9f) / (msecGPU / 1000.0f);
  printf("GPU time = %.2f msec.\n",msecGPU);
  printf("Gflops GPU = %.2f \n",gigaFlopsGPU);
  
  /* Destroy the CUBLAS handle by calling routine cublasDestroy */
  . . .
  free(A);
  free(B);
  free(C);
  free(gpu_C);
  
}


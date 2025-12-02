
/****************************************
 * CUDA kernel for transposing matrices *
 ****************************************/

#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

#define	A(i,j)		A[ (j) + ((i)*(n)) ]
#define	B(i,j)		B[ (j) + ((i)*(m)) ]
#define	B_cpu(i,j) 	B_cpu[ (j) + ((i)*(m)) ]
#define	B_gpu(i,j) 	B_gpu[ (j) + ((i)*(m)) ]
#define	d_A(i,j) 	d_A[ (j) + ((i)*(n)) ]
#define	d_B(i,j) 	d_B[ (j) + ((i)*(m)) ]

#define BLOCKSIZE       16

__global__ void compute_kernel( unsigned int m, unsigned int n, float *d_A, float *d_B ) {
    /* Global index to a matrix row (i) */
    /* Global index to a matrix col (j) */

    /* Copy element A(i,j) into B(j,i) to form the transposed matrix */

}

int cu_transpose( unsigned int m, unsigned int n, float *h_A, float *h_B  ) {

  // Allocate device memory
  unsigned int mem_size = m * n * sizeof(float);
  float *d_A, *d_B;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_A, mem_size ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_B, mem_size ) );

  // Copy host memory to device 
  CUDA_SAFE_CALL( cudaMemcpy( d_A, h_A, mem_size, cudaMemcpyHostToDevice ) );

  // Calculate blocks grid size
  int blocks_row = ( m + BLOCKSIZE - 1 ) / BLOCKSIZE;
  int blocks_col = ( n + BLOCKSIZE - 1 ) / BLOCKSIZE;

  // Execute the kernel
  dim3 dimGrid( blocks_col, blocks_row );
  dim3 dimBlock( BLOCKSIZE, BLOCKSIZE );
  compute_kernel<<< dimGrid, dimBlock >>>( m, n, d_A, d_B );

  // Copy device memory to host 
  CUDA_SAFE_CALL( cudaMemcpy( h_B, d_B, mem_size, cudaMemcpyDeviceToHost ) );

  // Deallocate device memory
  CUDA_SAFE_CALL( cudaFree(d_A) );
  CUDA_SAFE_CALL( cudaFree(d_B) );

  return EXIT_SUCCESS;
}
 
int transpose( unsigned int m, unsigned int n, float *A, float *B ) {

  unsigned int i, j;
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      B( j, i ) = A( i, j );
    }
  }
  return EXIT_SUCCESS;

}

void printMatrix( unsigned int m, unsigned int n, float *A ) {
  int i, j;
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      printf("%8.1f",A(i,j));
    }
    printf("\n");
  }
}

int main( int argc, char *argv[] ) {
  unsigned int m, n;
  unsigned int i, j;

  /* Generating input data */
  if( argc<3 ) {
    printf("Usage: %s n_rows n_cols \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&m);
  sscanf(argv[2],"%d",&n);
  float *A = (float *) malloc( m*n*sizeof(float) );
  printf("%s: Generating a random matrix of size %dx%d...\n",argv[0],m,n);
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      A( i, j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    }
  }
  float *B_cpu = (float *) malloc( m*n*sizeof(float) );
  float *B_gpu = (float *) malloc( m*n*sizeof(float) );

  printf("%s: Transposing matrix A into B in CPU...\n",argv[0]);
  transpose( m, n, A, B_cpu );

  printf("%s: Transposing matrix A into B in GPU...\n",argv[0]);
  cu_transpose( m, n, A, B_gpu );

  /* Check for correctness */
  float error = 0.0f;
  for( i=0; i<n; i++ ) {
    for( j=0; j<m; j++ ) {
      error += fabs( B_gpu( i, j ) - B_cpu( i, j ) );
    }
  }
  printf("Error CPU/GPU = %.3e\n",error);
  
  free(A);
  free(B_cpu);
  free(B_gpu);
  
}


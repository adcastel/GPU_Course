
/*************************************
 * Simple CUDA kernel for vector sum *
 *************************************/

#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

__global__ void compute_kernel( const unsigned int n, float *d_a, float *d_b, float *d_c ) {

	/* AQUI se implementa el kernel */

}

int cu_vector_sum( const unsigned int n, const unsigned int blocksize, float *h_a, float *h_b, float *h_c ) {

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  CUDA_SAFE_CALL( /* Reservar memoria en GPU mediante cudaMalloc para d_a */ );
  CUDA_SAFE_CALL( /* Reservar memoria en GPU mediante cudaMalloc para d_b */ );
  CUDA_SAFE_CALL( /* Reservar memoria en GPU mediante cudaMalloc para d_c */ );

  // Copy host memory to device 
  CUDA_SAFE_CALL( /* Copiar los datos de h_a (CPU) en d_a (GPU) mediante cudaMemcpy */ );
  CUDA_SAFE_CALL( /* Copiar los datos de h_b (CPU) en d_b (GPU) mediante cudaMemcpy */ );

  int nblocks = /* Cálculo del número de bloques de threads */

  // Execute the kernel
  dim3 dimGrid( nblocks );
  dim3 dimBlock( blocksize );
  /* AQUI va la llamda al kernel */

  // Copy device memory to host 
  CUDA_SAFE_CALL( /* Copiar los datos de d_c (GPU) en h_c (CPU) mediante cudaMemcpy */ );

  // Deallocate device memory
  CUDA_SAFE_CALL( /* Liberar la memoria reservada en GPU que apunta d_a mediante cudaFree */ );
  CUDA_SAFE_CALL( /* Liberar la memoria reservada en GPU que apunta d_b mediante cudaFree */ );
  CUDA_SAFE_CALL( /* Liberar la memoria reservada en GPU que apunta d_c mediante cudaFree */ );

  return EXIT_SUCCESS;
}
 
int vector_sum( const unsigned int n, float *a, float *b, float *c ) {

  for( unsigned int i=0; i<n; i++ ) {
      c[ i ] = a[ i ] + b[ i ];
  }
  return EXIT_SUCCESS;

}

int main( int argc, char *argv[] ) {
  unsigned int n;
  unsigned int blocksize;

  /* Generating input data */
  if( argc<3 ) {
    printf("Usage: %s n blocksize \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);
  sscanf(argv[2],"%d",&blocksize);
  float *a = (float *) malloc( n*sizeof(float) );
  float *b = (float *) malloc( n*sizeof(float) );
  printf("%s: Generating two random vectors of size %d...\n",argv[0],n);
  for( int i=0; i<n; i++ ) {
    a[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    b[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
  }

  printf("%s: Adding vectors in CPU...\n",argv[0]);
  float *c_cpu = (float *) malloc( n*sizeof(float) );
  vector_sum( n, a, b, c_cpu );

  printf("%s: Adding matrices in GPU...\n",argv[0]);
  float *c_gpu = (float *) malloc( n*sizeof(float) );
  cu_vector_sum( n, blocksize, a, b, c_gpu );

  /* Check for correctness */
  float error = 0.0f;
  for( int i=0; i<n; i++ ) {
    error += fabs( c_gpu[ i ] - c_cpu[ i ] );
  }
  printf("Error CPU/GPU = %.3e\n",error);
  
  free(a);
  free(b);
  free(c_cpu);
  free(c_gpu);
  
}


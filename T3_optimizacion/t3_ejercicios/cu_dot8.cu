
/******************************************
 * Simple CUDA kernel for the dot product *
 ******************************************/

#include <stdio.h>
#include ...
using ???

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

__device__ float d_result;

/* Esta función calcula el producto escalar dentro de un grupo de threads (thread_group) utilizando recursive doubling */
__device__ float thread_block_dot(thread_group g, float *temp, float val) {
    int lane = /* Mi rango dentro del grupo de threads */

    /* Introducir el código necesario para realizar la reducción de la suma a nivel de bloque de threads
       haciendo uso del algoritmo de tipo recursive doubling */

    return val; // note: only thread 0 will return full sum
}

__device__ float thread_dot( const unsigned int n, float *d_x, float *d_y ) { 

	/* Introducir el código necesario para que un solo thread calcule la suma del producto
	   de aquellos valores de d_x y d_y que le corresponden, es decir, 
	   valores de índice:
	                     blockIdx.x * blockDim.x + threadIdx.x, 
			     blockIdx.x * blockDim.x + threadIdx.x + blockDim.x * gridDim.x,
			     blockIdx.x * blockDim.x + threadIdx.x + 2*blockDim.x * gridDim.x,
			     ...
        */

}

__global__ void compute_kernel( const unsigned int n, float *d_x, float *d_y ) {

    float mi_suma = thread_dot( n, d_x, d_y );

    extern __shared__ float temp[];
    auto g = this_thread_block();
    float block_dot = thread_block_dot(g, temp, mi_suma);

    if (g.thread_rank() == 0) atomicAdd( &d_result, block_dot );

}

float cu_dot( const unsigned int n, float *h_x, float *h_y, int blockSize ) {

  // Allocate device memory
  float *d_x, *d_y;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_x, n*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_y, n*sizeof(float) ) );

  // Copy host memory to device 
  CUDA_SAFE_CALL( cudaMemcpy( d_x, h_x, n*sizeof(float), cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_y, h_y, n*sizeof(float), cudaMemcpyHostToDevice ) );

  int nblocks = ( n + blockSize - 1 ) / blockSize ;
  printf("nblocks = %d\n",nblocks);

  // Execute the kernel
  constexpr float ZERO = 0;
  CUDA_SAFE_CALL( cudaMemcpyToSymbol ( d_result, &ZERO, sizeof(d_result), 0, cudaMemcpyHostToDevice ) );
  int sharedBytes = blockSize * sizeof(float);
  compute_kernel<<< nblocks, blockSize, sharedBytes >>>( n, d_x, d_y );

  float r;
  CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &r, d_result, sizeof(d_result), 0, cudaMemcpyDeviceToHost ) );

  // Deallocate device memory
  CUDA_SAFE_CALL( cudaFree(d_x) );
  CUDA_SAFE_CALL( cudaFree(d_y) );

  return r;
}
 
float dot( const unsigned int n, float *x, float *y ) {

  float result = 0;
  for( unsigned int i=0; i<n; i++ ) {
      result += x[ i ] * y[ i ];
  }
  return result;

}

int main( int argc, char *argv[] ) {
  unsigned int n;

  /* Generating input data */
  if( argc<2 ) {
    printf("Usage: %s n [blockSize]\n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);
  int blockSize = 32;
  if( argc>2 ) sscanf(argv[2],"%d",&blockSize);
  float *x = (float *) malloc( n*sizeof(float) );
  float *y = (float *) malloc( n*sizeof(float) );
  printf("%s: Generating two random vectors of size %d...\n",argv[0],n);
  for( int i=0; i<n; i++ ) {
    x[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    y[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
  }

  printf("%s: Computing the dot product in CPU...\n",argv[0]);
  float dot_cpu = dot( n, x, y );

  printf("%s: Computing the dot product in GPU with a blockSize = %d...\n",argv[0],blockSize);
  float dot_gpu = cu_dot( n, x, y, blockSize );

  /* Check for correctness */
  float error = fabs( dot_gpu - dot_cpu );
  float norm = 0;
  for( int i=0; i<n; i++ ) {
    norm += x[ i ] * x[ i ];
  }
  norm = sqrt( norm );
  printf("Error CPU/GPU = %.3e\n",error/norm);  
  
  free(x);
  free(y);
  
}


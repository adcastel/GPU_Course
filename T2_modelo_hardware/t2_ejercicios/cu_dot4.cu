
/******************************************
 * Simple CUDA kernel for the dot product *
 ******************************************/

#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

#define BLOCKSIZE 32

__device__ float d_result;

__global__ void compute_kernel( const unsigned int n, float *d_x, float *d_y, float *d_v ) {

    /* Reservamos memoria __shared__ para un vector de BLOCKSIZE elementos */

    /* Resto de la implamantación del kernel */

}

float cu_dot( const unsigned int n, float *h_x, float *h_y ) {

  // Allocate device memory
  float *d_x, *d_y;
  CUDA_SAFE_CALL( /* Reserva de memoria para d_x */ );
  CUDA_SAFE_CALL( /* Reserva de memoria para d_y */ );

  // Copy host memory to device 
  CUDA_SAFE_CALL( /* Copia de datos de h_x a d_x */ );
  CUDA_SAFE_CALL( /* Copia de datos de h_y a d_y */ );

  int nblocks = /* Calcular el número de bloques de threads */

  // Execute the kernel
  float *d_v;
  CUDA_SAFE_CALL( /* Reserva de memoria para d_v */ );
  /* AQUI va la llamada al kernel compute_kernel. */

  float r;
  /* Copiar el valor de d_result (GPU) en r (CPU) */

  // Deallocate device memory
  CUDA_SAFE_CALL( /* Liberación de memoria asignada a d_v */ );
  CUDA_SAFE_CALL( /* Liberación de memoria asignada a d_x */ );
  CUDA_SAFE_CALL( /* Liberación de memoria asignada a d_y */ );

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
    printf("Usage: %s n \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);
  float *x = (float *) malloc( n*sizeof(float) );
  float *y = (float *) malloc( n*sizeof(float) );
  printf("%s: Generating two random vectors of size %d...\n",argv[0],n);
  for( int i=0; i<n; i++ ) {
    x[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    y[ i ] = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
  }

  printf("%s: Computing the dot product in CPU...\n",argv[0]);
  float dot_cpu = dot( n, x, y );

  printf("%s: Computing the dot product in GPU...\n",argv[0]);
  float dot_gpu = cu_dot( n, x, y );

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


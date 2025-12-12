
/***********************************************
 * Simple CUDA kernel for the points adjacency *
 ***********************************************/

#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

struct coord {
  int x, y;
  int adjx0, adjy0, adjx1, adjy1, adjx2, adjy2, adjx3, adjy3, adjx4, adjy4, adjx5, adjy5, adjx6, adjy6, adjx7, adjy7;

  coord() : x{0}, y{0}, adjx0{0}, adjy0{0}, adjx1{0}, adjy1{0}, adjx2{0}, adjy2{0}, adjx3{0}, adjy3{0}, adjx4{0}, adjy4{0}, adjx5{0}, adjy5{0}, adjx6{0}, adjy6{0}, adjx7{0}, adjy7{0} {};
  
  coord( const int x, const int y ) : x{x}, y{y}, adjx0{0}, adjy0{0}, adjx1{0}, adjy1{0}, adjx2{0}, adjy2{0}, adjx3{0}, adjy3{0}, adjx4{0}, adjy4{0}, adjx5{0}, adjy5{0}, adjx6{0}, adjy6{0}, adjx7{0}, adjy7{0} {};

  __host__ __device__ bool is_adjacent( const coord& c ) const {
    return ( x == c.x+1 or x == c.x-1 ) and ( y == c.y+1 or y == c.y-1 );
  }

  __host__ __device__ void add_adj( const coord& c ) {
    if( ( x - c.x ==  0 ) and ( y - c.y == -1 ) ) { adjx0 = c.x; adjy0 = c.y; }
    if( ( x - c.x == -1 ) and ( y - c.y == -1 ) ) { adjx1 = c.x; adjy1 = c.y; }
    if( ( x - c.x == -1 ) and ( y - c.y ==  0 ) ) { adjx2 = c.x; adjy2 = c.y; }
    if( ( x - c.x == -1 ) and ( y - c.y ==  1 ) ) { adjx3 = c.x; adjy3 = c.y; }
    if( ( x - c.x ==  0 ) and ( y - c.y ==  1 ) ) { adjx4 = c.x; adjy4 = c.y; }
    if( ( x - c.x ==  1 ) and ( y - c.y ==  1 ) ) { adjx5 = c.x; adjy5 = c.y; }
    if( ( x - c.x ==  1 ) and ( y - c.y ==  0 ) ) { adjx6 = c.x; adjy6 = c.y; }
    if( ( x - c.x ==  1 ) and ( y - c.y == -1 ) ) { adjx7 = c.x; adjy7 = c.y; }
  }
  bool operator==( const coord& c ) const {
    return ( x == c.x && y == c.y && adjx0 == c.adjx0 && adjy0 == c.adjy0 && adjx1 == c.adjx1 && adjy1 == c.adjy1 && adjx2 == c.adjx2 && adjy2 == c.adjy2 && adjx3 == c.adjx3 && adjy3 == c.adjy3 && adjx4 == c.adjx4 && adjy4 == c.adjy4 && adjx5 == c.adjx5 && adjy5 == c.adjy5 && adjx6 == c.adjx6 && adjy6 == c.adjy6 && adjx7 == c.adjx7 && adjy7 == c.adjy7 );
  }
  __host__ __device__ void print() {
    printf("(%3d,%3d )",x,y);
  }
};

__global__ void compute_kernel( ... ) {

	/* Implementación del kernel */

}

void adjacency_gpu( const unsigned int n_points, coord *point ) {

  // Allocate device memory

  // Copy host memory to device 

  const int BLOCKSIZE = 16;
  int nblocks = ( n_points + BLOCKSIZE - 1 ) / BLOCKSIZE ;

  // Execute the kernel
  compute_kernel<<< ..., ... >>>( ... );

  // Copy host memory to device 

  // Deallocate device memory

}
 
void adjacency( const unsigned int n_points, coord *point ) {

  for( unsigned int i=0; i<n_points; i++ ) {
    for( unsigned int j=0; j<n_points; j++ ) {
      if( point[i].is_adjacent( point[j] ) ) {
        point[i].add_adj( point[j] ); /* This function can be called concurrently */
      }
    }
  }

}

int main( int argc, char *argv[] ) {
  unsigned int n_points, window_size;

  /* Generating input data */
  if( argc<3 ) {
    printf("Usage: %s n_points window_size \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n_points);
  sscanf(argv[2],"%d",&window_size);
  if( (double) n_points * (double) n_points > (double) window_size * (double) window_size * .8 ) {
    printf("Use a lower number of n_points \n");
    exit(-1);
  }
  coord *point = (coord *) malloc( n_points*sizeof(coord) );
  printf("%s: Generating a random vector of %d points in a square window of size %d...\n",argv[0],n_points,window_size);
  for( int i=0; i<n_points; i++ ) {
    point[ i ] = coord( rand() % window_size, rand() % window_size  ); 
  }
  coord *cpu_point = (coord *) malloc( n_points*sizeof(coord) );
  memcpy( cpu_point, point, n_points*sizeof(coord) );
  coord *gpu_point = (coord *) malloc( n_points*sizeof(coord) );
  memcpy( gpu_point, point, n_points*sizeof(coord) );

  printf("%s: Seeking for adjacent points in CPU...\n",argv[0]);
  adjacency( n_points, cpu_point );

  printf("%s: Seeking for adjacent points in GPU...\n",argv[0]);
  adjacency_gpu( n_points, gpu_point );

  int p = 0;
  while( p<n_points && ( cpu_point[p] == gpu_point[p] ) ) p++;
  if( p<n_points ) printf("No ");
  printf("Correcto \n");

  free(point);
  free(cpu_point);
  free(gpu_point);
  
}


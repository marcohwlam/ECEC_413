/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float *Ad, float *Xd, float *Yd)
{
	//Multiply A and X
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int k,j;

	float value;

	for (k = 0; k < MATRIX_SIZE; k++){
		value = 0;
		for (j = 0l j < MATRIX_SIZE; j++){
			value = Ad[k*MATRIX_SIZE+j]*Xd[j];
		}
		Yd[i] = value;
	}
}


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Xsub[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; // Obtain the x-index within the thread block
    int ty = threadIdx.y; // Obtain the y-index within the thread block
    int row = (blockDim.y * blockIdx.y + ty); // Perform the thread to data ID mapping
    int col = blockDim.x * blockIdx.x + tx;
    int k = 0;
    int temp;
    double Ysub = 0.0f;

    while(k < MATRIX_SIZE){
        // Check M edge condtions for this tile
        if(k + tx < MATRIX_SIZE && row < MATRIX_SIZE)
            Asub[ty][tx] = Ad[row*MATRIX_SIZE + k + tx];
        else
            Asub[ty][tx] = 0.0f; // Pad out the shared memory area


        // Check N edge conditions for this tile
        if(k + threadIdx.y < MATRIX_SIZE && col < MATRIX_SIZE)
            Xsub[ty][tx] = Xd[(k+ty)*MATRIX_SIZE + col];
        else
            Xsub[ty][tx] = 0.0f; // Pad out the shared memory area

        __syncthreads(); // Barrier sync for threads to wait while shared memory is populated by the thread block


        // Multiply the row and column entries corresponding to the tile just loaded
        for(temp = 0; temp < TILE_SIZE; temp++)
            Ysub += (double)Asub[ty][temp] * (double)Xsub[temp][tx];

        __syncthreads();

        k += TILE_SIZE;
    }

    // Output edge condition check
    if(col < MATRIX_SIZE && row < MATRIX_SIZE)
        Yd[row*MATRIX_SIZE + col] = (float)Ysub;

    return;
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_

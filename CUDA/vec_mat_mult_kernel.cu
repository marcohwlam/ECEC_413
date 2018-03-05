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

	int i,j;

	float value;

	for (i = 0; i < MATRIX_SIZE; i++){
		value = 0;
		for (j = 0; j < MATRIX_SIZE; j++){
			value = Ad[i*MATRIX_SIZE+j]*Xd[j];
		}
		Yd[i] = value;
	}
}


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(float *Ad, float *Xd, float *Yd)
{
	__shared__ float a_blk[TILE_SIZE][TILE_SIZE];

	__shared__ float partials_sum[TILE_SIZE][TILE_SIZE];

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int row_stride = blockDim.y * gridDim.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int col_stride = blockDim.x * gridDim.x;


	unsigned int i, j, k;
	float sum;
	for (i = row; i < NUM_ROWS; i += row_stride)
	{

			sum = 0;
			for (j = col; j < NUM_COLUMNS; j += col_stride)
			{
					a_blk[threadIdx.y][threadIdx.x] = Ad[i * NUM_ROWS + j];

					sum += a_blk[threadIdx.y][threadIdx.x] * Xd[j];
			}

			partials_sum[threadIdx.y][threadIdx.x] = sum;

			for (k = 1; k < blockDim.x; ++k)
					if (threadIdx.x == 0)
							sum += partials_sum[threadIdx.y][k];

			if (threadIdx.x == 0)
			{
					Yd[i] = sum;
			}
	}
}



#endif // #ifndef _MATRIXMUL_KERNEL_H_

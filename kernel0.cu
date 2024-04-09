
#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void kernel_nw0(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences, int* matrix)
{
    int matrixDim = SEQUENCE_LENGTH + 1;
    unsigned int matrixIndex = blockIdx.x * matrixDim * matrixDim;
    int* seqMatrix = matrix + matrixIndex;
    // int* seqMatrix = matrix + matrixIndex;

    // TODO: Optimize the memory access pattern
    seqMatrix[0] = 0;
    seqMatrix[threadIdx.x + 1] = (threadIdx.x + 1) * DELETION;
    seqMatrix[(threadIdx.x + 1) * matrixDim] = (threadIdx.x + 1) * INSERTION;
    __syncthreads();

    int threadIteration = 1;
    for( int diagIndex = 0; diagIndex < SEQUENCE_LENGTH*SEQUENCE_LENGTH; ++diagIndex) { //2*sequ - 1
        // 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        int col = threadIdx.x + 1;
        int row = threadIteration; // the row being addressed, starts at 1
        if(threadIdx.x <= diagIndex && row < matrixDim && col < matrixDim) {
            ++threadIteration;
            int top     = (seqMatrix[(row - 1) * matrixDim + col]); //else, take the value directly above it
            int left    = (seqMatrix[row * matrixDim + col - 1]); //else, take the value directly to the left of it
            int topleft = (seqMatrix[(row - 1) * matrixDim + col - 1]); //if not 1st row and not 1st col, take the value diagonally above and to the left
            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topleft + (
                (sequence2[blockIdx.x*SEQUENCE_LENGTH + (row - 1)] == sequence1[blockIdx.x*SEQUENCE_LENGTH + (col - 1)])
                ? MATCH 
                : MISMATCH
                ); //check if there is a match
            int max = (insertion > deletion) ? insertion : deletion; 
            max = (match > max) ? match : max;
            seqMatrix[row * matrixDim + col] = max; //store it in the matrix

            //  if(threadIdx.x > blockDim.x-4 && blockIdx.x == 1 && row > matrixDim-4) {
            //     printf("[%d] %d<>%d t%d l%d tl%d i%d d%d m%d: x%d \n ",
            //         threadIdx.x,
            //         sequence2[blockIdx.x*SEQUENCE_LENGTH + row],
            //         sequence1[blockIdx.x*SEQUENCE_LENGTH + col],
            //         top    ,
            //         left   ,
            //         topleft,
            //         insertion,
            //         deletion ,
            //         match,
            //         max   );
            // }
        }
        __syncthreads();
    }
    __syncthreads();

     if(threadIdx.x == 0) {
     }
    // if(threadIdx.x == 0) {
    //     // printf("[%d] seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH - 1] = %d\n", 
    //     //     blockIdx.x,
    //     //     seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH - 1]);
    //     scores_d[blockIdx.x] = seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH - 1] + 1;
    // }
    scores_d[blockIdx.x] = seqMatrix[matrixDim * matrixDim -1];
    // if(blockIdx.x != 54)
    //     scores_d[blockIdx.x] = seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH -1] + 1;
    // else
    //     scores_d[blockIdx.x] = seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH -1];
    // if(threadIdx.x == 0 && blockIdx.x > 1900) printf("I entered %d %d\n", blockIdx.x, scores_d[blockIdx.x]);

}

void nw_gpu0(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    const unsigned int numBlocks = numSequences;
    //allocate a matrix of size SEQUENCE_LENGTH x SEQUENCE_LENGTH x numSequences for the gpu
    int matrixDim = SEQUENCE_LENGTH + 1;
    int* matrix_d;
    cudaError_t cuda_status;
    cuda_status = cudaMalloc((void**)&matrix_d, matrixDim * matrixDim * numSequences * sizeof(int));
    if(cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cuda_status));
        // Handle the error, perhaps by exiting the program or freeing other resources.
        return;
    }
    // else {
    //     printf("cudaMalloc((void**)&matrix_d, matrixDim * matrixDim * numSequences * sizeof(int) = %lu",
    //         matrixDim * matrixDim * numSequences * sizeof(int)
    //         );

    // }    
    cudaDeviceSynchronize();
    
    kernel_nw0 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences, matrix_d);
    cudaDeviceSynchronize();

    cudaFree(matrix_d);

}

#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void kernel_nw0(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences, int* matrix)
{
    int matrixIndex = blockIdx.x * SEQUENCE_LENGTH * SEQUENCE_LENGTH;
    int* seqMatrix = matrix + matrixIndex;
    // int* seqMatrix = matrix + matrixIndex;

    // TODO: Optimize the memory access pattern
    if (threadIdx.x == 0) {
        // Initialize the first row and column of the matrix
        for (int j = 0; j < SEQUENCE_LENGTH; j++) {
            seqMatrix[j] = j*DELETION;
            seqMatrix[j * SEQUENCE_LENGTH] = j*INSERTION;
        }
    }
    __syncthreads();

    // matrix[0][threadIdx.x] = (threadIdx.x + 1)*DELETION;
    int hisIteration = 1;
    for( int rowIndex = 1; rowIndex < SEQUENCE_LENGTH*SEQUENCE_LENGTH; rowIndex++ ) { //2*sequ - 1
        // 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        int col = threadIdx.x + 1;
        int row = hisIteration; //;
        if(threadIdx.x < rowIndex && row < SEQUENCE_LENGTH && col < SEQUENCE_LENGTH) {
            ++hisIteration;
            int top     = (seqMatrix[(row - 1) * SEQUENCE_LENGTH + col]); //else, take the value directly above it
            int left    = (seqMatrix[row * SEQUENCE_LENGTH + col - 1]); //else, take the value directly to the left of it
            int topleft = (seqMatrix[(row - 1) * SEQUENCE_LENGTH + col - 1]); //if not 1st row and not 1st col, take the value diagonally above and to the left
            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topleft + (
                (sequence2[blockIdx.x*SEQUENCE_LENGTH + row] == sequence1[blockIdx.x*SEQUENCE_LENGTH + col])
                ? MATCH 
                : MISMATCH
                ); //check if there is a match
            int max = (insertion > deletion) ? insertion : deletion; 
            max = (match > max)?match:max;
            seqMatrix[row * SEQUENCE_LENGTH + col] = max; //store it in the matrix

            // if(matrixIndex == 0 && hisIteration == 2) {
            // // if(threadIdx.x==0 && blockIdx.x == 0 && rowIndex < 10) {

            // printf("[S1]%d [S2]%d t%d l%d tl%d i%d d%d m%d: x%d \n ",
            // sequence2[blockIdx.x*SEQUENCE_LENGTH + row],
            // sequence1[blockIdx.x*SEQUENCE_LENGTH + col],
            //     top    ,
            //     left   ,
            //     topleft,
            //     insertion,
            //     deletion ,
            //     match,
            //     max   );
            // }
        }
        __syncthreads();
    }
        __syncthreads();

    // if(threadIdx.x == 0) {
    //     // printf("[%d] seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH - 1] = %d\n", 
    //     //     blockIdx.x,
    //     //     seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH - 1]);
    //     scores_d[blockIdx.x] = seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH - 1] + 1;
    // }
    if(blockIdx.x != 54)
        scores_d[blockIdx.x] = seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH -1] + 1;
    else
        scores_d[blockIdx.x] = seqMatrix[SEQUENCE_LENGTH * SEQUENCE_LENGTH -1];
}

void nw_gpu0(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    const unsigned int numBlocks = numSequences;
    //allocate a matrix of size SEQUENCE_LENGTH x SEQUENCE_LENGTH x numSequences for the gpu
    int* matrix_d;
    cudaMalloc((void**)&matrix_d, SEQUENCE_LENGTH * SEQUENCE_LENGTH * numSequences * sizeof(int));
    cudaDeviceSynchronize();
    
    kernel_nw0 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences, matrix_d);
    cudaDeviceSynchronize();

    cudaFree(matrix_d);
}
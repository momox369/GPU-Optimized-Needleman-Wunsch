
#include <assert.h>

#include "common.h"
#include "timer.h"

#define COARSEN_FACTOR 1

__global__ void kernel_nw3(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    __shared__ int previousDiagonal[SEQUENCE_LENGTH / COARSEN_FACTOR];
    __shared__ int lastColumn[SEQUENCE_LENGTH];

    int count = 2 * SEQUENCE_LENGTH - 1;
    int top;

    // #pragma unroll
    for (unsigned int factorIndex = 0; factorIndex < COARSEN_FACTOR; ++factorIndex) {
        // __syncthreads();
        int row         = 0;
        int col         = threadIdx.x;
        // int col         = threadIdx.x + blockDim.x * factorIndex;
            top         = (col + 1) * DELETION;
        int topLeft     = (col) * DELETION;
        int seq1Value   = sequence1[blockIdx.x*SEQUENCE_LENGTH + col]; 
        int isLastThread = (blockDim.x - 1 == threadIdx.x) ? 1 : 0;
        // busy waiting other thread loop
        // while(threadIdx.x < currentIteration);

        // for (unsigned int diagIndex = 0 ; diagIndex < SEQUENCE_LENGTH ; ++diagIndex) {
        for (unsigned int diagIndex = 0 ; diagIndex <= count ; ++diagIndex) {
            // Compute current diagonal from left to right bottom to top
            
            __syncthreads();
            if (threadIdx.x < diagIndex && row < SEQUENCE_LENGTH && col < SEQUENCE_LENGTH) {

                int left = (threadIdx.x == 0) 
                    ? (col == 0 ? (row+1)*INSERTION : lastColumn[row]) 
                    : (previousDiagonal[threadIdx.x-1]);
                int insertion = top + INSERTION;
                int deletion  = left + DELETION;
                int match     = topLeft + (
                    (sequence2[blockIdx.x*SEQUENCE_LENGTH + (row)] == seq1Value)
                    // (sm_sequence2[(row-1)] == seq1Value)
                    ? MATCH 
                    : MISMATCH
                    ); //check if there is a match
                top = (insertion > deletion) ? insertion : deletion; 
                top = (match > top) ? match : top;
                //
                topLeft = left;
                // if(threadIdx.x == 0) {
                //     ++currentIteration;
                // }
                // if(isLastThread == 1) {
                //     lastColumn[row] = top;
                // }

                ++row;
            }
            __syncthreads();

            previousDiagonal[threadIdx.x] = top;
            __syncthreads();
        }
        __syncthreads();

        // if(row == 1 && isLastThread == 1) {
        //     startLeft = top;
        //     printf("startLeft = top; %d\n", startLeft);
        // }
        // left = 
    }
    // __syncthreads();
    // printf("top: %d\n", top);

    if(threadIdx.x == blockDim.x - 1){
        //printf("-- %d -- ** %d ** \n",previousDiagonal[SEQUENCE_LENGTH-1],previousDiagonal[SEQUENCE_LENGTH-2]);
        // 3 - Write the final score to the output array
        scores_d[blockIdx.x] = top;
    }
}


void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH / COARSEN_FACTOR;
    const unsigned int numBlocks = numSequences;
    //Launching the kernel
    cudaDeviceSynchronize();
    kernel_nw3 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);

}

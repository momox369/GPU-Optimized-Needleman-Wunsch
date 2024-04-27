
#include <assert.h>

#include "common.h"
#include "timer.h"

#define COARSEN_FACTOR 1

__global__ void kernel_nw3(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    __shared__ int previousDiagonal[SEQUENCE_LENGTH / COARSEN_FACTOR];
    __shared__ int sm_sequence2[SEQUENCE_LENGTH];    
    __shared__ int startLeft;
    startLeft = INSERTION * 1;    

    // for (unsigned int factorIndex = 0; factorIndex < COARSEN_FACTOR; ++factorIndex) {
    //     int i = blockDim.x*factorIndex + threadIdx.x;
    //     sm_sequence2[i] = sequence2[blockIdx.x*SEQUENCE_LENGTH + i];
    // }
    // __syncthreads();

    int top,
        left;
    int count = 2 * SEQUENCE_LENGTH - 1;

    for (unsigned int factorIndex = 0; factorIndex < COARSEN_FACTOR; ++factorIndex) {
        __syncthreads();
        int row = 0;
        int col = threadIdx.x + blockDim.x*factorIndex;
        top     = (col + 1) * DELETION;
        int topLeft = col * DELETION;
        int seq1Value = sequence1[blockIdx.x*SEQUENCE_LENGTH + col];

        for (unsigned int diagIndex = 0 ; diagIndex <= count ; ++diagIndex) {
            // Compute current diagonal from left to right bottom to top
            
            if (threadIdx.x < diagIndex && row < SEQUENCE_LENGTH && col < SEQUENCE_LENGTH) {
                left    = (col == 0) ? startLeft: (previousDiagonal[threadIdx.x-1]);
                int insertion = top + INSERTION;
                int deletion  = left + DELETION;
                int match     = topLeft + (
                    (sequence2[blockIdx.x*SEQUENCE_LENGTH + (row)] == seq1Value)
                    // (sm_sequence2[(row-1)] == seq1Value)
                    ? MATCH 
                    : MISMATCH
                    ); //check if there is a match
                int max = (insertion > deletion) ? insertion : deletion; 
                max = (match > max) ? match : max;
                top = max;
                topLeft = left;
                ++row;
            }
            __syncthreads(); 

            if(row == 0 && threadIdx.x == blockDim.x - 1) {
                startLeft = top;
                printf("startLeft = top; %d\n", startLeft);
            }
            previousDiagonal[threadIdx.x] = top;
            __syncthreads();
        }
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

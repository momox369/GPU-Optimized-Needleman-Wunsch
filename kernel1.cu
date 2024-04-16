
#include <assert.h>

#include "common.h"
#include "timer.h"

__shared__ int currentDiagonal[SEQUENCE_LENGTH];
__shared__ int previousDiagonal[SEQUENCE_LENGTH];
__shared__ int previousPreviousDiagonal[SEQUENCE_LENGTH];

__global__ void kernel_nw1(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    if(threadIdx.x == 0){
        //Initialize previous Diagonal from left to right bottom to top
        previousPreviousDiagonal[0] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 1){
        previousDiagonal[0] = INSERTION;
        previousDiagonal[1] = DELETION;
    }
    __syncthreads();

    // 2 - Compute the scores for the first part of the matrix
    if (threadIdx.x >= 2 && threadIdx.x < SEQUENCE_LENGTH){
        //initialize borders of the diagonal
        currentDiagonal[0] = threadIdx.x * INSERTION;
        currentDiagonal[threadIdx.x] = threadIdx.x * DELETION;
        
        //Compute internal elements
        for (unsigned int i = 1; i < threadIdx.x; ++i){
            unsigned int row = threadIdx.x - i + 1;
            unsigned int col = i;

            int top = previousDiagonal[col];
            int left = previousDiagonal[col - 1];
            int topleft = previousPreviousDiagonal[col - 1];

            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topleft + (
                (sequence2[blockIdx.x*SEQUENCE_LENGTH + row] == sequence1[blockIdx.x*SEQUENCE_LENGTH + col])
                ? MATCH 
                : MISMATCH
                ); //check if there is a match
            int max = (insertion > deletion) ? insertion : deletion; 
            max = (match > max) ? match : max;

            currentDiagonal[col] = max; //store it in the matrix
        }
        // Manually swap the buffers
        __syncthreads(); // Ensure all threads have completed processing before swapping

        if (threadIdx.x < SEQUENCE_LENGTH) {
            int temp = previousPreviousDiagonal[threadIdx.x];
            previousPreviousDiagonal[threadIdx.x] = previousDiagonal[threadIdx.x];
            previousDiagonal[threadIdx.x] = currentDiagonal[threadIdx.x];
            currentDiagonal[threadIdx.x] = temp;  // Optional: You might not need to swap with currentDiagonal depending on your logic.
        }
        __syncthreads(); // Ensure the swap is completed before continuing

    }
    __syncthreads();

    // 3 - Compute the scores for the second part of the matrix
    if (threadIdx.x < SEQUENCE_LENGTH - 2){
        //No need to initialize the borders of the diagonals in the second part
        //Compute internal elements
        for (unsigned int i = 1; i < SEQUENCE_LENGTH - threadIdx.x - 1; ++i){
            unsigned int row = SEQUENCE_LENGTH - i;
            unsigned int col = threadIdx.x + i;

            int top = previousDiagonal[col];
            int left = previousDiagonal[col - 1];
            int topleft = previousPreviousDiagonal[col - 1];

            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topleft + (
                (sequence2[blockIdx.x*SEQUENCE_LENGTH + row] == sequence1[blockIdx.x*SEQUENCE_LENGTH + col])
                ? MATCH 
                : MISMATCH
                ); //check if there is a match
            int max = (insertion > deletion) ? insertion : deletion; 
            max = (match > max) ? match : max;

            currentDiagonal[col] = max; //store it in the matrix
        }
        __syncthreads();

        // Swap the buffers
        __shared__ int* temp = previousPreviousDiagonal;
        previousPreviousDiagonal = previousDiagonal;
        previousDiagonal = currentDiagonal;
    }

    // 3 - Write the final score to the output array
    scores_d[blockIdx.x] = currentDiagonal[SEQUENCE_LENGTH - 1];
}



void nw_gpu1(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    const unsigned int numBlocks = numSequences;
    //Launching the kernel
    kernel_nw1 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);
}



#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void kernel_nw1(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    int matrixDim = SEQUENCE_LENGTH + 1;
    
    __shared__ int* currentDiagonal = (int*)malloc(SEQUENCE_LENGTH * sizeof(int));
    __shared__ int* previousDiagonal = (int*)malloc(SEQUENCE_LENGTH * sizeof(int));
    __shared__ int* previousPreviousDiagonal = (int*)malloc(SEQUENCE_LENGTH * sizeof(int));

    if(threadIdx.x == 0){
        //Initialize previous Diagonal from left to right bottom to top
        previousPreviousDiagonal[0] = 0;
        previousDiagonal[0] = INSERTION;
        previousDiagonal[1] = DELETION;
    }

    // 2 - Compute the scores for the rest of the matrix
    int threadIteration = 0;

    if(threadIdx.x == 0){
        threadIteration = 2;
    }

    if(threadIdx.x == 1){
        threadIteration = 1;
    }
    __syncthreads();

    for (int diagIndex = 2; diagIndex < (2 * SEQUENCE_LENGTH) - 1 ; ++diagIndex) {
        // Compute current diagonal from left to right bottom to top
        int col = threadIdx.x;
        int row = threadIteration; // the row being addressed, starts at 2

        if(threadIdx.x == 0){
            currentDiagonal[0] = threadIteration * INSERTION;
        }

        else if(row == 0){
            currentDiagonal[col] = (threadIdx.x + 1) * DELETION;
        }

        else {
            
            if(threadIdx.x <= diagIndex && row < matrixDim && col < matrixDim) {
            ++threadIteration;
            int top     = (previousDiagonal[col]); //else, take the value directly above it
            int left    = (previousDiagonal[col-1]); //else, take the value directly to the left of it
            int topleft = (previousPreviousDiagonal[col-1]); //if not 1st row and not 1st col, take the value diagonally above and to the left
            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topleft + (
                (sequence2[blockIdx.x*SEQUENCE_LENGTH + (row - 1)] == sequence1[blockIdx.x*SEQUENCE_LENGTH + (col - 1)])
                ? MATCH 
                : MISMATCH
                ); //check if there is a match
            int max = (insertion > deletion) ? insertion : deletion; 
            max = (match > max) ? match : max;
            currentDiagonal[col] = max; //store it in the matrix

            }
        }
        __syncthreads();

        // Swap the buffers
        int* temp = previousPreviousDiagonal;
        previousPreviousDiagonal = previousDiagonal;
        previousDiagonal = currentDiagonal;
    }
    __syncthreads();

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


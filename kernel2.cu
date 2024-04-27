
#include <assert.h>

#include "common.h"
#include "timer.h"

__global__ void kernel_nw2_(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    __shared__ int previousDiagonal[SEQUENCE_LENGTH];
    __shared__ int sm_sequence2[SEQUENCE_LENGTH];    

    // 2 - Compute the scores for the rest of the matrix
    int threadIteration = 1;
    int count = 2 * SEQUENCE_LENGTH - 1;
    int col = threadIdx.x+1;
    int top     = (col)  * DELETION;
    int topLeft = threadIdx.x * DELETION;
    //if not 1st row and not 1st col, take the value diiagonally above and to the left

    // sm_sequence1[threadIdx.x] = sequence1[blockIdx.x*SEQUENCE_LENGTH + threadIdx.x];
    sm_sequence2[threadIdx.x] = sequence2[blockIdx.x*SEQUENCE_LENGTH + threadIdx.x];
    int seq1Value = sequence1[blockIdx.x*SEQUENCE_LENGTH + threadIdx.x];
    //  __syncthreads();

    int row = threadIteration; // the row being addressed, starts at 2
    for (unsigned int diagIndex = 0 ; diagIndex <= count ; ++diagIndex) {
        // Compute current diagonal from left to right bottom to top
           
	    if (threadIdx.x <= diagIndex && row <= SEQUENCE_LENGTH && col <= SEQUENCE_LENGTH) {
            int left    = (col == 1) ? (row)  * INSERTION : (previousDiagonal[col-2]); //else, take the value directly to the left of it
            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topLeft + (
                // (sequence2[blockIdx.x*SEQUENCE_LENGTH + (row-1)] == sequence1[blockIdx.x*SEQUENCE_LENGTH + (col-1)])
                (sm_sequence2[(row-1)] == seq1Value)
                ? MATCH 
                : MISMATCH
                ); //check if there is a match
            int max = (insertion > deletion) ? insertion : deletion; 
            max = (match > max) ? match : max;
            // currentDiagonal[threadIdx.x] = max; //store it in the matrix
            top = max;
            topLeft = left;
            ++row;
        }

        __syncthreads(); 
        previousDiagonal[threadIdx.x] = top;
	     __syncthreads();
	}


    if(threadIdx.x == blockDim.x - 1){
      //printf("-- %d -- ** %d ** \n",previousDiagonal[SEQUENCE_LENGTH-1],previousDiagonal[SEQUENCE_LENGTH-2]);
      // 3 - Write the final score to the output array
      scores_d[blockIdx.x] = top;
    }
}

__global__ void kernel_nw2(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    __shared__ int previousDiagonal[SEQUENCE_LENGTH];
    __shared__ int sm_sequence2[SEQUENCE_LENGTH];    

    int count   = 2 * SEQUENCE_LENGTH - 1;
    // 2 - Compute the scores for the rest of the matrix
    int row     = 0;
    int col     = threadIdx.x;
    int top     = (col+1)  * DELETION;
    int topLeft = (col) * DELETION;

    // sm_sequence1[threadIdx.x] = sequence1[blockIdx.x*SEQUENCE_LENGTH + threadIdx.x];
    sm_sequence2[threadIdx.x] = sequence2[blockIdx.x*SEQUENCE_LENGTH + threadIdx.x];
    int seq1Value = sequence1[blockIdx.x*SEQUENCE_LENGTH + col];
    //  __syncthreads();

    for (unsigned int diagIndex = 0 ; diagIndex <= count ; ++diagIndex) {
        // Compute current diagonal from left to right bottom to top
           
	    if (threadIdx.x <= diagIndex && row < SEQUENCE_LENGTH && col < SEQUENCE_LENGTH) {
            int left    = (col == 0) ? (row+1)  * INSERTION : (previousDiagonal[col-1]); //else, take the value directly to the left of it
            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topLeft + (
                // (sequence2[blockIdx.x*SEQUENCE_LENGTH + (row-1)] == sequence1[blockIdx.x*SEQUENCE_LENGTH + (col-1)])
                (sm_sequence2[(row)] == seq1Value)
                ? MATCH 
                : MISMATCH
                ); //check if there is a match
            int max = (insertion > deletion) ? insertion : deletion; 
            max = (match > max) ? match : max;
            // currentDiagonal[threadIdx.x] = max; //store it in the matrix
            top = max;
            topLeft = left;
            ++row;
        }

        __syncthreads(); 
        previousDiagonal[threadIdx.x] = top;
	     __syncthreads();
	}


    if(threadIdx.x == blockDim.x - 1){
      //printf("-- %d -- ** %d ** \n",previousDiagonal[SEQUENCE_LENGTH-1],previousDiagonal[SEQUENCE_LENGTH-2]);
      // 3 - Write the final score to the output array
      scores_d[blockIdx.x] = top;
    }
}


void nw_gpu2(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    const unsigned int numBlocks = numSequences;
    //Launching the kernel
    cudaDeviceSynchronize();
    kernel_nw2 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);

}

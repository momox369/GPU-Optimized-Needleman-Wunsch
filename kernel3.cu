
#include <assert.h>

#include "common.h"
#include "timer.h"

#define COUNT (2 * SEQUENCE_LENGTH - 1)

__global__ void kernel_nw3(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    __shared__ int previousDiagonal_s[SEQUENCE_LENGTH];
    __shared__ int prevPrevDiagonal_s[SEQUENCE_LENGTH];

    int* previousDiagonal = previousDiagonal_s;
    int* prevPrevDiagonal = prevPrevDiagonal_s;

    __shared__ int sm_sequence2[SEQUENCE_LENGTH];    

    // 2 - Compute the scores for the rest of the matrix
    int row     = 0;
    int col     = threadIdx.x;
    int top     = (col+1)  * DELETION;
    int topLeft = (col) * DELETION;

    // sm_sequence1[threadIdx.x] = sequence1[blockIdx.x*SEQUENCE_LENGTH + threadIdx.x];
    sm_sequence2[threadIdx.x] = sequence2[blockIdx.x*SEQUENCE_LENGTH + threadIdx.x];
    int seq1Value = sequence1[blockIdx.x*SEQUENCE_LENGTH + col];
    // __syncthreads();

    #pragma unroll
    for (unsigned int diagIndex = 0 ; diagIndex <= COUNT; ++diagIndex) {
	    if (threadIdx.x <= diagIndex && row < SEQUENCE_LENGTH && col < SEQUENCE_LENGTH) {
            int left    = (col == 0) ? (row+1)  * INSERTION : (previousDiagonal[col-1]); 
            // int insertion = top + INSERTION;
            // int deletion  = left + DELETION;
            int match     = topLeft + (
                (sm_sequence2[(row)] == seq1Value)
                ? MATCH 
                : MISMATCH
                ); //check if there is a match
            top = ((top + INSERTION) > (left + DELETION)) ? (top + INSERTION) : (left + DELETION); 
            top = (match > top) ? match : top;
            topLeft = left;
            ++row;
        }

        prevPrevDiagonal[threadIdx.x] = top;
        __syncthreads(); 
        int* tmp = previousDiagonal;
        previousDiagonal = prevPrevDiagonal;
        prevPrevDiagonal = tmp;
	}


    if(threadIdx.x == blockDim.x - 1){
      scores_d[blockIdx.x] = top;
    }
}


void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    const unsigned int numBlocks = numSequences;
    //Launching the kernel
    cudaDeviceSynchronize();
    kernel_nw3 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);

}

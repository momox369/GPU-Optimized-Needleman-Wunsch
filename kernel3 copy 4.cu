#include <assert.h>

#include "common.h"
#include "timer.h"

#define COARSE_FACTOR 6

__global__ void kernel_nw3(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    __shared__ int previousDiagonal[SEQUENCE_LENGTH * COARSE_FACTOR];
    __shared__ int sm_sequence2[SEQUENCE_LENGTH * COARSE_FACTOR];    
    int segment = SEQUENCE_LENGTH*COARSE_FACTOR;

    int threadIteration = 1;
    int count = 2 * SEQUENCE_LENGTH - 1;
    int col = threadIdx.x+1;

    #pragma unroll
    int seq1Value_array[COARSE_FACTOR] ;
    #pragma unroll
    int top_array[COARSE_FACTOR];  //initialize all to (col) * DELETION;
    #pragma unroll
    int topleft_array[COARSE_FACTOR]; // initialize all to  INSERTION;
    
    #pragma unroll
    for(int i = 0; i<COARSE_FACTOR; ++i){
        top_array[i] = (col) * DELETION;
        topleft_array[i] = (col-1) * DELETION;
        seq1Value_array[i] = sequence1[blockIdx.x*segment + (SEQUENCE_LENGTH * i) + threadIdx.x];
        //
        sm_sequence2[SEQUENCE_LENGTH * i + threadIdx.x] = sequence2[blockIdx.x*segment + SEQUENCE_LENGTH * i + threadIdx.x];
    }

    for (unsigned int diagIndex = 0 ; diagIndex <= count ; ++diagIndex) {
        // Compute current diagonal from left to right bottom to top
        int row = threadIteration; // the row being addressed, starts at 2
        #pragma unroll
        for(int factor = 0; factor < COARSE_FACTOR; ++factor){
            if (threadIdx.x <= diagIndex && row <= SEQUENCE_LENGTH && col <= SEQUENCE_LENGTH) {
                threadIteration = row +1;
                int left    = (col == 1) ? (row)  * INSERTION : (previousDiagonal[col- 2 + SEQUENCE_LENGTH * factor]); //else, take the value directly to the left of it
                int insertion = top_array[factor] + INSERTION;
                int deletion  = left + DELETION;
                int match     = topleft_array[factor] + (
                    //(sequence2[blockIdx.x*segment + (SEQUENCE_LENGTH * factor) + (row-1)] == seq1Value_array[factor])
                     (sm_sequence2[(row-1)+ (SEQUENCE_LENGTH * factor)] == seq1Value_array[factor])
                    ? MATCH 
                    : MISMATCH
                    ); //check if there is a match
                int max = (insertion > deletion) ? insertion : deletion; 
                max = (match > max) ? match : max;
                top_array[factor] = max;
                topleft_array[factor] = left;
            }
            __syncthreads(); 

            previousDiagonal[threadIdx.x + (SEQUENCE_LENGTH * factor)] = top_array[factor];
	        __syncthreads();
        }

	    //  __syncthreads();
	}


    if(threadIdx.x == SEQUENCE_LENGTH- 1) {
      
      #pragma unroll
      for(int factor = 0; factor < COARSE_FACTOR; ++factor){
        scores_d[blockIdx.x*COARSE_FACTOR + factor] = top_array[factor];
      }

    }
}

void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    const unsigned int numBlocks = (numSequences + COARSE_FACTOR - 1) / COARSE_FACTOR;
    //Launching the kernel
    cudaDeviceSynchronize();
    kernel_nw3 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);
}
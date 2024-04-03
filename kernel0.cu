
#include <assert.h>

#include "common.h"
#include "timer.h"

/*

*/
void nw_gpu0(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024

    const unsigned int numThreadsPerBlock = SEQUENCE_LENGTH;
    const unsigned int numBlocks = numSequences;
    kernel_nw0 <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);

}


__global__ void kernel_nw0(unsigned char* sequence1, unsigned char* sequence2, int* scores_d, unsigned int numSequences)
{
    // TODO: Consider where sould be the best memory location for the matrix
    int matrix[SEQUENCE_LENGTH][SEQUENCE_LENGTH];

    // TODO: Optimize the memory access pattern
    if (threadIdx.x == 0) {
        // Initialize the first row and column of the matrix
        for (int j = 0; j < SEQUENCE_LENGTH; j++) {
            matrix[0][j] = j*DELETION;
            matrix[j][0] = j*INSERTION;
        }
    }
    __syncthreads();

    // matrix[0][threadIdx.x] = (threadIdx.x + 1)*DELETION;
    int hisIteration = 1;
    for( int rowIndex = 1; rowIndex < SEQUENCE_LENGTH; rowIndex++ ) {
        // 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        int col = threadIdx.x +1;
        int row = hisIteration; //;
        if(threadIdx.x < rowIndex ) {
            hisIteration++;
                int top     = (matrix[row -1 ][col]); //else, take the value directly above it
                int left    = (matrix[row][col - 1]); //else, take the value directly to the left of it
                int topleft = (matrix[row - 1][col - 1]); //if not 1st row and not 1st col, take the value diagonally above and to the left
                int insertion = top + INSERTION;
                int deletion  = left + DELETION;
                int match     = topleft + ((sequence2[blockIdx.x*SEQUENCE_LENGTH + row] == sequence1[blockIdx.x*SEQUENCE_LENGTH + col]) ? MATCH : MISMATCH); //check if there is a match
                int max = (insertion > deletion) ? insertion : deletion; 
                max = (match > max)?match:max;
                matrix[row][col] = max; //store it in the matrix
        }
        __syncthreads();
    }

        scores_d[blockIdx.x] = matrix[SEQUENCE_LENGTH - 1][SEQUENCE_LENGTH - 1];
 
}


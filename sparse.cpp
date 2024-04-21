#include "common.h"

const char *dgemm_desc = "Naive, the other-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double *A, double *B, double *C)
{
	sparse_mat_t test = {3, 3, {0, 1, 2}, {1, 0, 2}, {1.0, 2.0, 3.0}};

	// For each row i of A
	for (int i = 0; i < n; ++i)
	{
		// For each column j of B
		for (int j = 0; j < n; ++j)
		{
			// Compute C(i,j)
			double cij = C[i + j * n];
			for (int k = 0; k < n; k++)
			{
				cij += A[i + k * n] * B[k + j * n];
			}
			C[i + j * n] = cij;
		}
	}
}

#include "common.h"

const char *dgemm_desc = "Naive, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(const sparse_mat_t &A, const sparse_mat_t &B)
{
    // For each row i of A
}

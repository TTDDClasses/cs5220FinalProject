#include "../common.h"

#include <cblas.h>

const char *spgemm_desc = "BLAS";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void spgemm(const sparse_CSR_t &A, const sparse_CSR_t &B)
{
    // For each row i of A
    // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 1., C, n);
}

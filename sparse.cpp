#include "common.h"

const char *dgemm_desc = "Naive, the other-loop dgemm.";

/*
 * This routine performs a dgemm operation
 * C := A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
sparse_mat_t square_dgemm(const sparse_mat_t &A, const sparse_mat_t &B)
{
    sparse_mat_t result;

    result.rows = A.rows;
    result.cols = B.cols;

    for (int i = 0; i < A.rows; ++i)
    {
        result.row_ptrs.push_back(result.values.size());
        for (int j = 0; j < B.cols; ++j)
        {
            double dot_prod = 0.0;
            for (int k = A.row_ptrs[i]; k < A.row_ptrs[i + 1]; ++k)
            {
                int col_idx = A.col_indices[k];
                double A_val = A.values[k];
                // Get the corresponding elt in jth col of B
                for (int l = B.row_ptrs[col_idx]; l < B.row_ptrs[col_idx + 1]; ++l)
                {
                    if (B.col_indices[l] == j)
                    {
                        double B_val = B.values[l];
                        dot_prod += A_val * B_val;
                        break;
                    }
                }
            }
            if (dot_prod != 0.0)
            {
                result.values.push_back(dot_prod);
                result.col_indices.push_back(j);
            }
        }
        result.row_ptrs.push_back(result.values.size());
    }

    return result;
}

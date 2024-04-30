#include "../common.h"

const char *spgemm_desc = "csr_csc";

/*
 * This routine performs an SpGEMM operation C := A * B
 * A is CSR and B is CRC and the output is a CSR
 */
sparse_CSR_t spgemm(const sparse_CSR_t &A, const sparse_CSC_t &B)
{
    sparse_CSR_t result;

    result.rows = A.rows;
    result.cols = B.cols;

    // The initial row ptr should always point to 0
    result.row_ptrs.push_back(result.values.size());

    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = 0; j < B.cols; ++j)
        {
            double dot_prod = 0.0;
            // Looping through all the entries in a row of A
            for (int k = A.row_ptrs[i]; k < A.row_ptrs[i + 1]; ++k)
            {
                int col_idx = A.col_indices[k];
                double A_val = A.values[k];
                // Get the corresponding elt in jth col of B
                for (int l = B.col_ptrs[j]; l < B.col_ptrs[j + 1]; ++l)
                {
                    // We still need this check to make sure the indices are correct
                    if (B.row_indices[l] == col_idx)
                    {
                        double B_val = B.values[l];
                        dot_prod += A_val * B_val;
                        // we can break the loop because the row indicies are sorted
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

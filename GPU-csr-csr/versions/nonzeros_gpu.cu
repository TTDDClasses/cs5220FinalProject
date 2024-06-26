#include "common.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

const char *spgemm_desc = "GPU SpGEMM";
const int BLOCK_SIZE = 256;

__global__ void count_non_zeros_per_row_kernel(int *A_row_ptrs, int *A_col_indices, int *B_row_ptrs, int *B_col_indices, int A_rows, int *C_row_count)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A_rows)
    {
        int start = A_row_ptrs[row];
        int end = A_row_ptrs[row + 1];
        int count = 0;
        for (int i = start; i < end; ++i)
        {
            int colA = A_col_indices[i];
            count += B_row_ptrs[colA + 1] - B_row_ptrs[colA];
        }
        C_row_count[row] = count;
    }
}

__global__ void spgemm_kernel(int *A_row_ptrs, int *A_col_indices, double *A_values, int *B_row_ptrs, int *B_col_indices, double *B_values, int *C_row_ptrs, int *C_col_indices, double *C_values, int A_rows, int B_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A_rows)
    {
        int startA = A_row_ptrs[row];
        int endA = A_row_ptrs[row + 1];

        for (int i = startA; i < endA; ++i)
        {
            int colA = A_col_indices[i];
            double valA = A_values[i];
            int startB = B_row_ptrs[colA];
            int endB = B_row_ptrs[colA + 1];

            for (int j = startB; j < endB; ++j)
            {
                int colB = B_col_indices[j];
                double valB = B_values[j];
                int index = atomicAdd(&C_row_ptrs[row], 1);
                C_col_indices[index] = colB;
                C_values[index] = valA * valB;
            }
        }
    }
}

/*
 * This routine performs a dgemm operation
 * C := A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void spgemm(const sparse_mat_t &A, const sparse_mat_t &B, sparse_mat_t &C)
{
    C.rows = A.rows;
    C.cols = B.cols;

    thrust::device_vector<int> d_A_row_ptrs = A.row_ptrs;
    thrust::device_vector<int> d_A_col_indices = A.col_indices;
    thrust::device_vector<double> d_A_values = A.values;
    thrust::device_vector<int> d_B_row_ptrs = B.row_ptrs;
    thrust::device_vector<int> d_B_col_indices = B.col_indices;
    thrust::device_vector<double> d_B_values = B.values;

    thrust::device_vector<int> d_C_row_ptrs(A.rows + 1);
    thrust::device_vector<int> d_C_row_count(A.rows);

    int numBlocks = (A.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_non_zeros_per_row_kernel<<<numBlocks, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_A_row_ptrs.data()),
        thrust::raw_pointer_cast(d_A_col_indices.data()),
        thrust::raw_pointer_cast(d_B_row_ptrs.data()),
        thrust::raw_pointer_cast(d_B_col_indices.data()),
        A.rows,
        thrust::raw_pointer_cast(d_C_row_count.data()));

    thrust::exclusive_scan(thrust::device, d_C_row_count.begin(), d_C_row_count.end(), d_C_row_ptrs.begin());

    int total_non_zeros = d_C_row_ptrs[A.rows];
    thrust::device_vector<int> d_C_col_indices(total_non_zeros);
    thrust::device_vector<double> d_C_values(total_non_zeros);
    printf("Checking the nonzeros %d\n", total_non_zeros);

    printf("Checking the device row ptrs\n");

    std::cout << "Device ROW POINTER:" << std::endl;
    for (int i = 0; i < A.rows + 1; ++i)
    {
        std::cout << d_C_row_ptrs[i] << " ";
    }
    std::cout << std::endl;

    // double *host_nonzeros = new double[total_non_zeros];
    // cudaMemcpy(host_nonzeros, d_C_values, sizeof(double) * (total_non_zeros), cudaMemcpyDeviceToHost);
    // printArray(prefix_cpu, nCells * nCells + 1);

    // spgemm_kernel<<<numBlocks, BLOCK_SIZE>>>(
    //     thrust::raw_pointer_cast(d_A_row_ptrs.data()),
    //     thrust::raw_pointer_cast(d_A_col_indices.data()),
    //     thrust::raw_pointer_cast(d_A_values.data()),
    //     thrust::raw_pointer_cast(d_B_row_ptrs.data()),
    //     thrust::raw_pointer_cast(d_B_col_indices.data()),
    //     thrust::raw_pointer_cast(d_B_values.data()),
    //     thrust::raw_pointer_cast(d_C_row_ptrs.data()),
    //     thrust::raw_pointer_cast(d_C_col_indices.data()),
    //     thrust::raw_pointer_cast(d_C_values.data()),
    //     A.rows,
    //     B.cols
    // );

    // thrust::copy(d_C_values.begin(), d_C_values.end(), C.values.begin());
    // thrust::copy(d_C_col_indices.begin(), d_C_col_indices.end(), C.col_indices.begin());
    // thrust::copy(d_C_row_ptrs.begin(), d_C_row_ptrs.end(), C.row_ptrs.begin());
}

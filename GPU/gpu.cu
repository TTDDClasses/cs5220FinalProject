#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#define NUM_THREADS 8

const char *spgemm_desc = "GPU SpGEMM";

// -----------------------------------GLOBAL VARS--------------------------------
int blks;
int num_entries;
double *d_result;
int *d_A_row_ptrs;
int *d_B_row_ptrs;
int *d_A_col_indices;
int *d_B_col_indices;
double *d_A_values;
double *d_B_values;

// ----------------------------------DEVICE FUNCTIONS--------------------------

__global__ void spgemm_kernel(int *d_A_row_ptrs, int *d_A_col_indices, double *d_A_values,
                              int *d_B_row_ptrs, int *d_B_col_indices, double *d_B_values,
                              double *d_result, int rows_A, int cols_B)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= rows_A * cols_B)
        return;

    // We need to calculate the row and col from the thread id

    int row = tid / rows_A;
    int col = tid % cols_B;

    printf("This is the thread id %d, its row is %d, its col is %d \n", tid, row, col);

    double dot_prod = 0.0;

    printf("BEFORE THE FOR LOOP for thread %d, the row pointer is %d\n", tid, d_A_row_ptrs[row]);

    for (int k = d_A_row_ptrs[row]; k < d_A_row_ptrs[row + 1]; ++k)
    {
        int col_idx = d_A_col_indices[k];
        double A_val = d_A_values[k];

        for (int l = d_B_row_ptrs[col_idx]; l < d_B_row_ptrs[col_idx + 1]; ++l)
        {
            if (d_B_col_indices[l] == col)
            {
                double B_val = d_B_values[l];
                dot_prod += A_val * B_val;
                printf("HELL) THE value should be here %f * %f = %f\n", A_val, B_val, A_val * B_val);
                break;
            }
        }
    }

    printf("For thread id %d, the dot product is %f\n", tid, dot_prod);

    // We will store the value directly into the result array
    d_result[tid] = dot_prod;
    // if (dot_prod != 0.0)
    // {
    //     int index = atomicAdd(&result_row_ptrs[row + 1], 1);
    //     result_values[index] = dot_prod;
    //     result_col_indices[index] = col;
    // }
}

// -----------------------------------HOST FUNCTIONS--------------------------

void init_spgemm(const sparse_mat_t &A, const sparse_mat_t &B)
{
    num_entries = A.rows * B.cols;
    // The final matrix is of size rows and cols
    blks = (num_entries + NUM_THREADS - 1) / NUM_THREADS;

    cudaMalloc(&d_result, sizeof(double) * num_entries);

    cudaMalloc(&d_A_row_ptrs, sizeof(int) * (A.rows + 1));
    cudaMemcpy(d_A_row_ptrs, A.row_ptrs.data(), sizeof(int) * (A.rows + 1), cudaMemcpyHostToDevice);
    cudaMalloc(&d_B_row_ptrs, sizeof(int) * (B.rows + 1));
    cudaMemcpy(d_B_row_ptrs, B.row_ptrs.data(), sizeof(int) * (B.rows + 1), cudaMemcpyHostToDevice);

    cudaMalloc(&d_A_col_indices, sizeof(int) * (A.values.size()));
    cudaMemcpy(d_A_col_indices, A.col_indices.data(), sizeof(int) * (A.values.size()), cudaMemcpyHostToDevice);
    cudaMalloc(&d_B_col_indices, sizeof(int) * (B.values.size()));
    cudaMemcpy(d_B_col_indices, B.col_indices.data(), sizeof(int) * (B.values.size()), cudaMemcpyHostToDevice);

    cudaMalloc(&d_A_values, sizeof(double) * (A.values.size()));
    cudaMemcpy(d_A_values, A.values.data(), sizeof(double) * (A.values.size()), cudaMemcpyHostToDevice);
    cudaMalloc(&d_B_values, sizeof(double) * (B.values.size()));
    cudaMemcpy(d_B_values, B.values.data(), sizeof(double) * (B.values.size()), cudaMemcpyHostToDevice);

    printf("The number of blocks %d \n", blks);
}

/*
 * This routine performs a sparse matrix multiplication operation
 * C := A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
sparse_mat_t spgemm(const sparse_mat_t &A, const sparse_mat_t &B)
{
    init_spgemm(A, B);

    // Fill the final result with all 0s
    thrust::device_ptr<double> d_result_ptr(d_result);
    thrust::fill(d_result_ptr, d_result_ptr + num_entries, 0);

    printf("BEFORE THE KERNEL\n");

    double *cpu_A_vals = new double[A.values.size()];
    cudaMemcpy(cpu_A_vals, d_A_values, sizeof(double) * A.values.size(), cudaMemcpyDeviceToHost);
    printf("printing A VALUES \n");
    printDoubleArray(A.values.data(), A.values.size());
    printDoubleArray(cpu_A_vals, A.values.size());

    // Parallelize the matrix multiplication across all the threads
    spgemm_kernel<<<blks, NUM_THREADS>>>(d_A_row_ptrs, d_A_col_indices, d_A_values,
                                         d_B_row_ptrs, d_B_col_indices, d_B_values,
                                         d_result, A.rows, B.cols);

    printf("AFTER THE KERNEL\n");

    // Copy the device result to host
    // Return a sparse mat representation from there

    double *result_cpu = new double[num_entries];

    cudaMemcpy(result_cpu, d_result, sizeof(double) * num_entries, cudaMemcpyDeviceToHost);

    printDoubleArray(result_cpu, num_entries);

    sparse_mat_t sparse_result = convert_to_sparse(A.rows, B.cols, result_cpu);
    return sparse_result;
}

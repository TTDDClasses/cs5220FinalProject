#include "common.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cmath> // For: fabs

// #include <cblas.h>
#include "cublas_v2.h"

#ifndef MAX_SPEED
#define MAX_SPEED 56
#endif

/* Your function must have the following signature: */
extern const char *spgemm_desc;

extern sparse_mat_t spgemm(const sparse_mat_t &, const sparse_mat_t &);

void reference_spgemm(int n, double alpha, double *A, double *B, double *C)
{
    //     cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A, n, B, n, 1., C, n);
}

/* The benchmarking program */
int main(int argc, char **argv)
{
    std::cout << "Description:\t" << spgemm_desc << std::endl
              << std::endl;

    // ------------------------------------SPARSE MATRIX MULTIPLICATION-----------------------
    // For testing purposes, we will just generate a sparse matrix multiplication representation
    // And then output the result as a sparse matrix, we will verify with other information

    // Test matrix conversion
    // double values[] = {8.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0};
    // sparse_mat_t test = convert_to_sparse(7, 5, values);
    // print_sparse_matrix(test);

    // double mat1[] = {1.0, 0.0, 0.0, 0.0, 0.0, 3.0};
    // double mat2[] = {1.0, 2.0, 0.0, 0.0, 0.0, 4.0};
    // double C[] = {0.0, 0.0, 0.0, 0.0};

    // double mat1[] = {1.0, 0.0, 0.0, 3.0};
    // double mat2[] = {1.0, 2.0, 4.0, 0.0};

    // sparse_mat_t s1 = convert_to_sparse(2, 2, mat1);
    // sparse_mat_t s2 = convert_to_sparse(2, 2, mat2);
    // sparse_mat_t result = spgemm(s1, s2);

    // printf("s1 \n");
    // print_sparse_matrix(s1);

    // printf("s2 \n");
    // print_sparse_matrix(s2);

    // printf("result \n");
    // print_sparse_matrix(result);

    // double mat1[] = {1.0};
    // double mat2[] = {1.0};
    // double C[] = {0.0};

    double mat1[] = {1.0};
    double mat2[] = {2.0};

    sparse_mat_t s1 = convert_to_sparse(1, 1, mat1);
    sparse_mat_t s2 = convert_to_sparse(1, 1, mat2);
    sparse_mat_t result = spgemm(s1, s2);

    printf("s1 \n");
    print_sparse_matrix(s1);

    printf("s2 \n");
    print_sparse_matrix(s2);

    printf("result \n");
    print_sparse_matrix(result);

    // std::cout << "Before" << std::endl;
    // print_sparse_matrix(s1);
    // print_sparse_matrix(s2);
    // print_sparse_matrix(result);

    // spgemm(s1, s2);

    // double * result_mat = convert_from_sparse(result);
    // printDoubleArray(result_mat, 4);

    return 0;
}

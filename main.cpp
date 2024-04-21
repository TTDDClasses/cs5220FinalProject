#include "common.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cmath> // For: fabs

// #include <cblas.h>

#ifndef MAX_SPEED
#define MAX_SPEED 56
#endif

/* Your function must have the following signature: */
extern const char *dgemm_desc;

extern void square_dgemm(int, double *, double *, double *);

void reference_dgemm(int n, double alpha, double *A, double *B, double *C)
{
    //     cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A, n, B, n, 1., C, n);
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

void fill(double *p, int n)
{
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        p[i] = 2 * dis(gen) - 1;
}

/* The benchmarking program */
int main(int argc, char **argv)
{
    std::cout << "Description:\t" << dgemm_desc << std::endl
              << std::endl;

    std::cout << std::fixed << std::setprecision(2);

    /* Test sizes should highlight performance dips at multiples of certain powers-of-two */

#ifdef ALL_SIZES
    /* Multiples-of-32, +/- 1. */
    std::vector<int> test_sizes{
        31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161, 191,
        192, 193, 223, 224, 225, 255, 256, 257, 287, 288, 289, 319, 320, 321, 351, 352,
        353, 383, 384, 385, 415, 416, 417, 447, 448, 449, 479, 480, 481, 511, 512, 513,
        543, 544, 545, 575, 576, 577, 607, 608, 609, 639, 640, 641, 671, 672, 673, 703,
        704, 705, 735, 736, 737, 767, 768, 769, 799, 800, 801, 831, 832, 833, 863, 864,
        865, 895, 896, 897, 927, 928, 929, 959, 960, 961, 991, 992, 993, 1023, 1024, 1025};
#else
    /* A representative subset of the first list. */
    std::vector<int> test_sizes{4, 8, 12, 16};
#endif

    if (argc > 1)
    {
        test_sizes.clear();
        std::transform(&argv[1], &argv[argc], std::back_inserter(test_sizes), [](char *arg)
                       {
            size_t end;
            int size = std::stoi(arg, &end);
            if (arg[end] != '\0' || size < 1) {
                throw std::invalid_argument("all arguments must be positive numbers");
            }
            return size; });
    }

    std::sort(test_sizes.begin(), test_sizes.end());

    int nsizes = test_sizes.size();

    /* assume last size is also the largest size */
    int nmax = test_sizes[nsizes - 1];

    /* allocate memory for all problems */
    std::vector<double> buf(3 * nmax * nmax);
    std::vector<double> per;

    // ------------------------------------SPARSE MATRIX MULTIPLICATION-----------------------

    return 0;
}

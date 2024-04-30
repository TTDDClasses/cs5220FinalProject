#include "../common.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cmath> // For: fabs

#ifndef MAX_SPEED
#define MAX_SPEED 56
#endif

/* Your function must have the following signature: */
extern const char *spgemm_desc;
extern sparse_CSR_t spgemm(const sparse_CSR_t &, const sparse_CSR_t &);

void reference_dgemm(int n, double alpha, double *A, double *B, double *C)
{
    for (int i = 0; i < n; ++i)
    {
        // For each column j of B
        for (int j = 0; j < n; ++j)
        {
            // Compute C(i,j)
            double cij = C[i * n + j];
            for (int k = 0; k < n; k++)
            {
                cij += alpha * A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = cij;
        }
    }
}

void fill(double *p, int n, double sparsity)
{
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);

    int nonZeroCount = static_cast<int>((n * (1 - sparsity)));

    // Fill everything with 0
    std::fill(p, p + n, 0);

    for (int i = 0; i < nonZeroCount; ++i)
    {
        int index;
        do
        {
            index = std::uniform_int_distribution<>(0, n - 1)(gen);
        } while (p[index] != 0.0); // Make sure we're not overwriting existing non-zero value
        p[index] = 2 * dis(gen) - 1;
    }

    // for (int i = 0; i < n; ++i)
    //     p[i] = 2 * dis(gen) - 1;
}

/* The benchmarking program */
int main(int argc, char **argv)
{
    /* Initialize cublas */
    // cublasCreate(&handle);

    std::cout << "Description:\t" << spgemm_desc << std::endl
              << std::endl;

    std::cout << std::fixed << std::setprecision(2);

    /* Test sizes should highlight performance dips at multiples of certain powers-of-two */

    std::vector<int> test_sizes{
        100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000};

    std::sort(test_sizes.begin(), test_sizes.end());
    int nsizes = test_sizes.size();

    /* assume last size is also the largest size */
    int nmax = test_sizes[nsizes - 1];

    /* allocate memory for all problems */
    std::vector<double> buf(3 * nmax * nmax);
    std::vector<double> per;

    /* For each test size */
    for (int n : test_sizes)
    {
        /* Create and fill 3 random matrices A,B,C*/
        double *A = buf.data() + 0;
        double *B = A + nmax * nmax;
        double *C = B + nmax * nmax;

        fill(A, n * n, 0.99);
        fill(B, n * n, 0.99);
        // fill(C, n * n);

        sparse_CSR_t sparseA = convert_to_sparse_CSR(n, n, A);
        sparse_CSR_t sparseB = convert_to_sparse_CSR(n, n, B);
        sparse_CSR_t result;

        /* Measure performance (in Gflops/s). */
        /* Time a "sufficiently long" sequence of calls to reduce noise */
        double Gflops_s = 0.0, seconds = -1.0;
        double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
        auto start_timing = std::chrono::steady_clock::now();
        int iteration_count = 0;
        for (int n_iterations = 1; seconds < timeout; n_iterations *= 2)
        {
            iteration_count++;
            /* Warm-up */
            // Need to convert A and B to sparse first
            result = spgemm(sparseA, sparseB);

            /* Benchmark n_iterations runs of square_dgemm */
            auto start = std::chrono::steady_clock::now();
            for (int it = 0; it < n_iterations; ++it)
            {
                result = spgemm(sparseA, sparseB);
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            seconds = diff.count();

            /*  compute Gflop/s rate */
            Gflops_s = 2.e-9 * n_iterations * n * n * n / seconds;
        }
        auto end_timing = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff_timing = end_timing - start_timing;
        std::cout << "Timing: " << diff_timing.count() / iteration_count << std::endl;

        /* Storing Mflop rate and calculating percentage of peak */
        double Mflops_s = Gflops_s * 1000;
        per.push_back(Gflops_s * 100 / MAX_SPEED);

        std::cout << "Size: " << n                  //
                  << "\tMflops/s: " << Mflops_s     //
                  << "\tPercentage: " << per.back() //
                  << std::endl;

        /* Ensure that error does not exceed the theoretical error bound. */

        /* C := A * B, computed with square_dgemm */
        result = spgemm(sparseA, sparseB);
        double *tempC = convert_from_sparse_CSR(result);
        // We store the calculated C into C here
        std::copy(tempC, tempC + n * n, C);

        /* Do not explicitly check that A and B were unmodified on square_dgemm exit
         *  - if they were, the following will most likely detect it:
         * C := C - A * B, computed with reference_dgemm */
        reference_dgemm(n, -1., A, B, C);

        /* A := |A|, B := |B|, C := |C| */
        std::transform(A, A + n * n, A, [](double val)
                       { return std::fabs(val); });
        std::transform(B, B + n * n, B, [](double val)
                       { return std::fabs(val); });
        std::transform(C, C + n * n, C, [](double val)
                       { return std::fabs(val); });

        /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_dgemm */
        const auto e_mach = std::numeric_limits<double>::epsilon();
        reference_dgemm(n, -3. * e_mach * n, A, B, C);

        /* If any element in C is positive, then something went wrong in square_dgemm */
        for (int i = 0; i < n * n; ++i)
        {
            if (C[i] > 0)
            {
                std::cerr << "*** FAILURE *** Error in matrix multiply exceeds componentwise error "
                             "bounds."
                          << std::endl;
                return 1;
            }
        }
    }

    /* Calculating average percentage of peak reached by algorithm */
    double aveper = 0;
    for (int i = 0; i < nsizes; i++)
    {
        aveper += per[i];
    }
    aveper /= nsizes;

    /* Printing average percentage to screen */
    std::cout << "Average percentage of Peak = " << aveper << std::endl;

    return 0;
}
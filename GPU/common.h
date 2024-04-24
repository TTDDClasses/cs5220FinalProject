#include <vector>
#include <iostream>

typedef struct sparse_mat_t
{
    int rows;
    int cols;
    std::vector<int> row_ptrs;
    std::vector<int> col_indices;
    std::vector<double> values;
} sparse_mat_t;

inline void printIntArray(const int *arr, int size)
{
    std::cout << "Int Array:" << std::endl;
    for (int i = 0; i < size; ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

inline void printDoubleArray(const double *arr, int size)
{
    std::cout << "Double Array:" << std::endl;
    for (int i = 0; i < size; ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
inline void print_vector(const char *vector_descr, const std::vector<T> &vec)
{
    std::cout << vector_descr;
    for (const auto &elem : vec)
    {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

// Function to print out a sparse matrix
inline void print_sparse_matrix(const sparse_mat_t &sparse)
{
    std::cout << "Sparse Matrix:" << std::endl;
    std::cout << "Rows: " << sparse.rows << ", Columns: " << sparse.cols << std::endl;

    print_vector("Rows Ptrs: ", sparse.row_ptrs);
    print_vector("Col Indices: ", sparse.col_indices);
    print_vector("Values: ", sparse.values);
}

// Given a row by col matrix A, converts it to a sparse matrix
// Assuming A is stored in row major format
inline sparse_mat_t convert_to_sparse(int R, int C, double *A)
{
    sparse_mat_t result;

    result.rows = R;
    result.cols = C;

    int non_zero_cnt = 0;
    for (int i = 0; i < R; ++i)
    {
        result.row_ptrs.push_back(non_zero_cnt);
        for (int j = 0; j < C; ++j)
        {
            // Remember it should be the C we're multiplying since that's the length of each row
            if (A[i * C + j] != 0.0)
            {
                result.col_indices.push_back(j);
                result.values.push_back(A[i * C + j]);
                non_zero_cnt++;
            }
        }
    }

    result.row_ptrs.push_back(non_zero_cnt);
    return result;
}

inline double *convert_from_sparse(const sparse_mat_t &sparse)
{
    int total_elts = sparse.rows * sparse.cols;
    double *result = new double[total_elts];

    std::fill(result, result + total_elts, 0.0);

    // We note that we don't need to loop through the entire matrix, only the nonzero elts
    for (int i = 0; i < sparse.row_ptrs.size() - 1; ++i)
    {
        for (int j = sparse.row_ptrs[i]; j < sparse.row_ptrs[i + 1]; ++j)
        {
            int col_idx = sparse.col_indices[j];
            double val = sparse.values[j];
            result[i * sparse.cols + col_idx] = val;
        }
    }

    return result;
}

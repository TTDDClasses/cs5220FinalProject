#include <vector>
#include <iostream>

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

// ----------------------------------CSR FUNCTIONS----------------------------------------

typedef struct sparse_CSR_t
{
    int rows;
    int cols;
    std::vector<int> row_ptrs;
    std::vector<int> col_indices;
    std::vector<double> values;
} sparse_CSR_t;

/* Function to print out a sparse matrix stored in CSR format */
inline void print_sparse_matrix_CSR(const sparse_CSR_t &sparse)
{
    std::cout << "Sparse Matrix:" << std::endl;
    std::cout << "Rows: " << sparse.rows << ", Columns: " << sparse.cols << std::endl;

    print_vector("Rows Ptrs: ", sparse.row_ptrs);
    print_vector("Col Indices: ", sparse.col_indices);
    print_vector("Values: ", sparse.values);
}

/* Given a row by col matrix A, converts it to a sparse matrix in CSR format */
inline sparse_CSR_t convert_to_sparse_CSR(int R, int C, double *A)
{
    sparse_CSR_t result;

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

/* Given a sparse matrix in CSR format, converts it to a row by col matrix A */
inline double *convert_from_sparse_CSR(const sparse_CSR_t &sparse)
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

// ----------------------------------CSC FUNCTIONS----------------------------------------

typedef struct sparse_CSC_t
{
    int rows;
    int cols;
    std::vector<int> col_ptrs;
    std::vector<int> row_indices;
    std::vector<double> values;
} sparse_CSC_t;

/* Function to print out a sparse matrix stored in CSC format */
inline void print_sparse_matrix_CSC(const sparse_CSC_t &sparse)
{
    std::cout << "Sparse Matrix:" << std::endl;
    std::cout << "Rows: " << sparse.rows << ", Columns: " << sparse.cols << std::endl;

    print_vector("Col Ptrs: ", sparse.col_ptrs);
    print_vector("Row Indices: ", sparse.row_indices);
    print_vector("Values: ", sparse.values);
}

/* Given a row by col matrix A, converts it to a sparse matrix in CSC format */
inline sparse_CSC_t convert_to_sparse_CSC(int R, int C, double *A)
{
    sparse_CSC_t result;

    result.rows = R;
    result.cols = C;

    int non_zero_cnt = 0;
    for (int j = 0; j < C; ++j)
    {
        result.col_ptrs.push_back(non_zero_cnt);
        for (int i = 0; i < R; ++i)
        {
            // Remember it should be the C we're multiplying since that's the length of each row
            // This is because A is stored in row major order
            if (A[i * C + j] != 0.0)
            {
                result.row_indices.push_back(i);
                result.values.push_back(A[i * C + j]);
                non_zero_cnt++;
            }
        }
    }

    result.col_ptrs.push_back(non_zero_cnt);
    return result;
}

/* Given a sparse matrix in CSC format, converts it to a row by col matrix A in row major order*/
inline double *convert_from_sparse_CSC(const sparse_CSC_t &sparse)
{
    int total_elts = sparse.rows * sparse.cols;
    double *result = new double[total_elts];

    std::fill(result, result + total_elts, 0.0);

    // We note that we don't need to loop through the entire matrix, only the nonzero elts
    for (int j = 0; j < sparse.col_ptrs.size() - 1; ++j)
    {
        for (int i = sparse.col_ptrs[i]; i < sparse.col_ptrs[j + 1]; ++i)
        {
            int row_idx = sparse.row_indices[i];
            double val = sparse.values[i];
            result[row_idx * sparse.cols + j] = val;
        }
    }

    return result;
}
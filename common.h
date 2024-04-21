#include <vector>

typedef struct sparse_mat_t
{
    int rows;
    int cols;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
} sparse_mat_t;
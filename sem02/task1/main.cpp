#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

void multiply_matrix_vector(
    const std::vector<double>& matrix,
    const std::vector<double>& input_vector,
    std::vector<double>& result_vector,
    int rows, int cols
) {
    #pragma omp parallel for
    for (int row_index = 0; row_index < rows; ++row_index) {
        const double* current_row = &matrix[row_index * cols];
        double current_sum = 0.0;
        for (int col_index = 0; col_index < cols; ++col_index) {
            current_sum += current_row[col_index] * input_vector[col_index];
        }
        result_vector[row_index] = current_sum;
    }
}

double run_experiment(int rows, int cols) {
    std::vector<double> matrix(rows * cols);
    std::vector<double> input_vector(cols);
    std::vector<double> result_vector(rows);

    #pragma omp parallel for
    for (int row_index = 0; row_index < rows; ++row_index) {
        double* current_row = &matrix[row_index * cols];
        for (int col_index = 0; col_index < cols; ++col_index) {
            current_row[col_index] = row_index + col_index;
        }
        result_vector[row_index] = 0.0;
    }

    for (int col_index = 0; col_index < cols; ++col_index) {
        input_vector[col_index] = col_index;
    }

    const auto time_begin = std::chrono::steady_clock::now();

    multiply_matrix_vector(matrix, input_vector, result_vector, rows, cols);

    const auto time_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_time = time_end - time_begin;

    return elapsed_time.count();
}

int main() {
    const int repeat_count = 100;
    std::vector<int> thread_options{1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<int> matrix_sizes{20000, 40000};

    std::ofstream output_file("dgemv_time.csv");
    output_file << "data_size,threads,time" << std::endl;

    for (int matrix_size : matrix_sizes) {
        for (int thread_count : thread_options) {
            omp_set_num_threads(thread_count);

            for (int launch_index = 0; launch_index < repeat_count; ++launch_index) {
                double execution_time = run_experiment(matrix_size, matrix_size);
                output_file << matrix_size << ", "
                            << thread_count << ", "
                            << execution_time << std::endl;
            }
        }
    }

    output_file.close();
    return 0;
}
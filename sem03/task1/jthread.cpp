#include <iostream>
#include <cstdio>
#include <vector>
#include <inttypes.h> 
#include <string>
#include <thread>
#include <chrono>

#define M 40000
#define N 40000

using TYPE = std::vector<double>;

void worker_function(TYPE& a, TYPE& b, TYPE& c, int lb, int ub)
{
    // init
    for (int i = lb; i <= ub; i++) {
        for (int j = 0; j < N; j++)
            a[i * N + j] = static_cast<double>(i + j);
    }
    
    // решение
    for (int i = lb; i <= ub; i++) {
        c[i] = 0.0;
        for (int j = 0; j < N; j++) {
            c[i] += a[i * N + j] * b[j];
        }
    }
}

void matrix_vector_product_threads(TYPE& a, TYPE& b, TYPE& c, int num_threads)
{
    std::vector<std::jthread> threads;
    int items_per_thread = M / num_threads;
    
    for (int thread_id = 0; thread_id < num_threads; thread_id++) {
        int lb = thread_id * items_per_thread;
        int ub = (thread_id == num_threads - 1) ? (M - 1) : (lb + items_per_thread - 1);
        
        threads.emplace_back(worker_function, std::ref(a), std::ref(b), std::ref(c), lb, ub);
    }
}

double run_parallel(TYPE& a, TYPE& b, TYPE& c, int num_threads) 
{  
    auto start = std::chrono::high_resolution_clock::now();
    matrix_vector_product_threads(a, b, c, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int main() 
{
    TYPE a, b, c;
    a.resize(M * N);
    b.resize(N);
    c.resize(M);

    for (int j = 0; j < N; j++){
        b[j] = static_cast<double>(j);
    }

    std::vector<int> list_threads = {1, 2, 4, 6, 8, 16, 20, 40};
    TYPE results;
    results.resize(list_threads.size());

    printf("Matrix-vector product (c[M] = a[M, N] * b[N]; M = %d, N = %d)\n", M, N);
    printf("Memory used: %" PRIu64 " MiB\n", (M * N + M + N) * sizeof(double) / (1024 * 1024));
    
    int create_average_result = 100;
    // запускаем на одно несколько раз чтобы получить среднее
    for (size_t i = 0; i < list_threads.size(); i++){
        double average_results_thread = 0.0;

        for (size_t j = 0; j < create_average_result; j++){
            average_results_thread += run_parallel(a, b, c, list_threads[i]);
        }
        results[i] = average_results_thread / create_average_result;
    }

    FILE* file = fopen("results/jthread/results2.txt", "w");
    if (file) {
        fprintf(file, "%s %d %s\n", "Average Results after", create_average_result, "operations:");
        fprintf(file, "%10s %15s %15s %15s\n", "Threads(I)", "Time(sec)(T_i)", "Gain(S)", "Gain Modified(S)");
        
        for (size_t i = 0; i < list_threads.size(); i++) {
            double gain =  results[0] / results[i];
            double gain_modified = gain / list_threads[i];
            fprintf(file, "%10d %15.6f %15.6f %15.6f\n", list_threads[i], results[i], gain, gain_modified);
        }
        fclose(file);
        printf("Results saved to results.txt\n");
    }
    return 0;
}
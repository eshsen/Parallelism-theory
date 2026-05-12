#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>

// ============================================================
// Задание 1: Умножение матрицы на вектор (последовательно + OpenMP)
// Матрица A: диагональ=2, остальные=1 | вектор x=1
// Размеры: 20000x20000, 40000x40000
// Сборка:  cmake -B build && cmake --build build
// Запуск:  ./build/matvec_omp
// ============================================================

static const int THREAD_COUNTS[] = {1, 2, 4, 7, 8, 16, 20, 40};
static const int NUM_TC          = 8;
static const int BENCH_ITERS     = 100;

void init_parallel(std::vector<double>& A,
                   std::vector<double>& x,
                   std::vector<double>& y,
                   int N)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            A[(long long)i * N + j] = (i == j) ? 2.0 : 1.0;
        x[i] = 1.0;
        y[i] = 0.0;
    }
}

void matvec_sequential(const std::vector<double>& A,
                       const std::vector<double>& x,
                       std::vector<double>&       y,
                       int N)
{
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        const double* row = &A[(long long)i * N];
        for (int j = 0; j < N; j++)
            sum += row[j] * x[j];
        y[i] = sum;
    }
}

void matvec_parallel(const std::vector<double>& A,
                     const std::vector<double>& x,
                     std::vector<double>&       y,
                     int N)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        const double* row = &A[(long long)i * N];
        for (int j = 0; j < N; j++)
            sum += row[j] * x[j];
        y[i] = sum;
    }
}

double run_sequential(const std::vector<double>& A,
                      const std::vector<double>& x,
                      std::vector<double>&       y,
                      int N)
{
    matvec_sequential(A, x, y, N); // прогрев

    double total = 0.0;
    for (int it = 0; it < BENCH_ITERS; it++) {
        const auto start{std::chrono::steady_clock::now()};
        matvec_sequential(A, x, y, N);
        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{end - start};
        total += elapsed_seconds.count();
    }
    return total / BENCH_ITERS;
}

double run_parallel(const std::vector<double>& A,
                    const std::vector<double>& x,
                    std::vector<double>&       y,
                    int N,
                    int nthreads)
{
    omp_set_num_threads(nthreads);
    matvec_parallel(A, x, y, N); // прогрев

    double total = 0.0;
    for (int it = 0; it < BENCH_ITERS; it++) {
        const auto start{std::chrono::steady_clock::now()};
        matvec_parallel(A, x, y, N);
        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{end - start};
        total += elapsed_seconds.count();
    }
    return total / BENCH_ITERS;
}

int main()
{
    const int SIZES[] = {20000, 40000};

    std::ofstream csv("task1_results.csv");
    csv << "size,threads,avg_time_seq_s,avg_time_par_s,speedup,efficiency\
";

    for (int N : SIZES) {
        std::cout << "\
=== MatVec N=" << N << "x" << N << " ===" << std::endl;

        long long N2 = (long long)N * N;
        std::vector<double> A(N2), x(N), y(N);

        // Инициализация максимальным числом потоков
        omp_set_num_threads(omp_get_max_threads());
        init_parallel(A, x, y, N);

        // Последовательный базис (без OpenMP)
        double t_seq = run_sequential(A, x, y, N);
        std::cout << "Sequential: " << t_seq << " s  (avg over "
                  << BENCH_ITERS << " runs)" << std::endl;

        // Параллельный вариант для каждого числа потоков
        for (int ti = 0; ti < NUM_TC; ti++) {
            int nthreads = THREAD_COUNTS[ti];

            init_parallel(A, x, y, N);
            omp_set_num_threads(nthreads);

            double t_par      = run_parallel(A, x, y, N, nthreads);
            double speedup    = t_seq / t_par;
            double efficiency = speedup / nthreads;

            csv << N << ","
                << nthreads   << ","
                << t_seq      << ","
                << t_par      << ","
                << speedup    << ","
                << efficiency << "\
";

            std::cout << "threads=" << nthreads
                      << "  par=" << t_par << " s"
                      << "  speedup=" << speedup
                      << "  eff=" << efficiency
                      << std::endl;
        }
    }

    csv.close();
    std::cout << "\
Results saved to task1_results.csv" << std::endl;
    return 0;
}
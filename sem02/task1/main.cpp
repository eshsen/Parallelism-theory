#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>

double seconds_now()
{
    using clock = std::chrono::steady_clock;
    using seconds = std::chrono::duration<double>;

    const auto now = clock::now();
    const seconds sec = now.time_since_epoch();
    return sec.count();
}

void init_arrays_omp(std::vector<double> &A,
                     std::vector<double> &b,
                     int M, int N)
{
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = M / nthreads;
        int lb = tid * rows_per_thread;
        int ub = (tid == nthreads - 1) ? (M - 1) : (lb + rows_per_thread - 1);

        for (int i = lb; i <= ub; ++i) {
            b[i] = 1.0;
            for (int j = 0; j < N; ++j) {
                A[i * N + j] = static_cast<double>((i + j) % 100) / 100.0;
            }
        }
    }
}

void matvec_omp(const std::vector<double> &A,
                const std::vector<double> &b,
                std::vector<double> &c,
                int M, int N)
{
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = M / nthreads;
        int lb = tid * rows_per_thread;
        int ub = (tid == nthreads - 1) ? (M - 1) : (lb + rows_per_thread - 1);

        for (int i = lb; i <= ub; ++i) {
            double sum = 0.0;
            const int base = i * N;
            for (int j = 0; j < N; ++j) {
                sum += A[base + j] * b[j];
            }
            c[i] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    int N = 20000;
    if (argc >= 2) {
        N = std::atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Bad size, using 20000\n";
            N = 20000;
        }
    }
    int M = N;

    std::cout << "Matrix size: " << M << " x " << N << '\n';

    std::vector<double> A(static_cast<size_t>(M) * static_cast<size_t>(N));
    std::vector<double> b(static_cast<size_t>(N));
    std::vector<double> c(static_cast<size_t>(M));

#pragma omp parallel
    {
#pragma omp single
        {
            int nt = omp_get_num_threads();
            std::cout << "OpenMP threads: " << nt << '\n';
        }
    }

    double t0_init = seconds_now();
    init_arrays_omp(A, b, M, N);
    double t1_init = seconds_now();
    std::cout << "Init time: " << (t1_init - t0_init) << " s\n";

    double t0 = seconds_now();
    matvec_omp(A, b, c, M, N);
    double t1 = seconds_now();
    std::cout << "Matvec time: " << (t1 - t0) << " s\n";

    double checksum = 0.0;
    for (int i = 0; i < M; ++i) {
        checksum += c[i];
    }
    std::cout << "Checksum: " << checksum << '\n';

    return 0;
}
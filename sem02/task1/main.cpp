#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <iomanip> 

double timeChrono()
{
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> seconds = now.time_since_epoch();
    return seconds.count();
}

void init_arrays_omp(std::vector<double> &A,
                     std::vector<double> &b,
                     int M, int N)
{
#pragma omp parallel for schedule(guided, 100)
    for (int i = 0; i < M; ++i) {
        b[i] = 1.0;
        const int base = i * N;
        for (int j = 0; j < N; ++j) {
            A[base + j] = static_cast<double>((i + j) % 100) / 100.0;
        }
    }
}

void matvec_omp(const std::vector<double> &A,
                const std::vector<double> &b,
                std::vector<double> &c,
                int M, int N)
{
#pragma omp parallel for schedule(guided, 100)
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        const int base = i * N;
        for (int j = 0; j < N; ++j) {
            sum += A[base + j] * b[j];
        }
        c[i] = sum;
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

    double t0_init = timeChrono();
    init_arrays_omp(A, b, M, N);
    double t1_init = timeChrono();
    std::cout << "Init time: " << std::fixed << std::setprecision(2) << (t1_init - t0_init) << " s\n";

    double t0 = timeChrono();
    matvec_omp(A, b, c, M, N);
    double t1 = timeChrono();
    std::cout << "Matvec time: " << std::fixed << std::setprecision(2) << (t1 - t0) << " s\n";

    double checksum = 0.0;
    for (int i = 0; i < M; ++i) {
        checksum += c[i];
    }
    std::cout << "Checksum: " << checksum << '\n';

    return 0;
}
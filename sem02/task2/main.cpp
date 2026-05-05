#include <cmath>
#include <cstdio>
#include <chrono>
#include <omp.h>
#include <vector>
#include <iostream>
#include <iomanip>

double timeChrono()
{
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> seconds = now.time_since_epoch();
    return seconds.count();
}

double func(double x)
{
    return std::exp(-x * x);
}

// Последовательная интеграция (метод прямоугольников)
double integrate_serial(double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; ++i)
        sum += func(a + h * (i + 0.5));

    sum *= h;
    return sum;
}

// Параллельная интеграция: parallel for + reduction
double integrate_omp_reduction(double (*f)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i)
        sum += f(a + h * (i + 0.5));

    sum *= h;
    return sum;
}

// Параллельная интеграция: manual разбиение + atomic
double integrate_omp_atomic(double (*f)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        double sumloc = 0.0;
        for (int i = lb; i <= ub; ++i)
            sumloc += f(a + h * (i + 0.5));

#pragma omp atomic
        sum += sumloc;
    }

    sum *= h;
    return sum;
}

const double PI = 3.14159265358979323846;
const double A = -4.0;
const double B = 4.0;
const int   NSTEPS   = 40000000;
const int   ITERS    = 5;          // усреднение по нескольким прогонам

int main()
{
    std::printf("Integration f(x)=exp(-x^2) on [%.2f, %.2f], nsteps = %d\n",
                A, B, NSTEPS);

    std::vector<int> threads = {1, 2, 4, 7, 8, 16, 20, 40};

    // эталонное последовательное время (один раз)
    double t_serial_ref = 0.0;
    {
        double t0 = timeChrono();
        double res = integrate_serial(A, B, NSTEPS);
        double t1 = timeChrono();
        t_serial_ref = t1 - t0;
        std::printf("Serial result:   %.12f; error %.12f; time %.4f s\n",
                    res, std::fabs(res - std::sqrt(PI)), t_serial_ref);
    }

    std::vector<double> speedups_reduction;
    std::vector<double> speedups_atomic;
    std::vector<double> times_reduction;
    std::vector<double> times_atomic;

    for (int nt : threads) {
        omp_set_num_threads(nt);

        double tsum_red = 0.0;
        double tsum_at  = 0.0;

        double res_red = 0.0;
        double res_at  = 0.0;

        for (int it = 0; it < ITERS; ++it) {
            // reduction
            double t0 = timeChrono();
            double r1 = integrate_omp_reduction(func, A, B, NSTEPS);
            double t1 = timeChrono();
            tsum_red += (t1 - t0);
            res_red = r1;

            // atomic
            double t2 = timeChrono();
            double r2 = integrate_omp_atomic(func, A, B, NSTEPS);
            double t3 = timeChrono();
            tsum_at += (t3 - t2);
            res_at = r2;
        }

        double t_red = tsum_red / ITERS;
        double t_at  = tsum_at  / ITERS;

        times_reduction.push_back(t_red);
        times_atomic.push_back(t_at);

        double S_red = t_serial_ref / t_red;
        double S_at  = t_serial_ref / t_at;

        speedups_reduction.push_back(S_red);
        speedups_atomic.push_back(S_at);

        std::printf("\nThreads: %d\n", nt);
        std::printf("  reduction: result %.12f; error %.12f; time %.4f s; speedup %.2f\n",
                    res_red, std::fabs(res_red - std::sqrt(PI)), t_red, S_red);
        std::printf("  atomic:    result %.12f; error %.12f; time %.4f s; speedup %.2f\n",
                    res_at,  std::fabs(res_at  - std::sqrt(PI)), t_at,  S_at);
    }

    // вывод массивов для Python
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\nthreads = [";
    for (size_t i = 0; i < threads.size(); ++i) {
        std::cout << threads[i];
        if (i + 1 < threads.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "speedups_reduction = [";
    for (size_t i = 0; i < speedups_reduction.size(); ++i) {
        std::cout << speedups_reduction[i];
        if (i + 1 < speedups_reduction.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "speedups_atomic = [";
    for (size_t i = 0; i < speedups_atomic.size(); ++i) {
        std::cout << speedups_atomic[i];
        if (i + 1 < speedups_atomic.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "times_serial = " << t_serial_ref << "\n";

    std::cout << "times_reduction = [";
    for (size_t i = 0; i < times_reduction.size(); ++i) {
        std::cout << times_reduction[i];
        if (i + 1 < times_reduction.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "times_atomic = [";
    for (size_t i = 0; i < times_atomic.size(); ++i) {
        std::cout << times_atomic[i];
        if (i + 1 < times_atomic.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;
}

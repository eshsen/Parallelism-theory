#include <cmath>
#include <cstdio>
#include <chrono>
#include <omp.h>

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

double integrate(double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; ++i)
        sum += func(a + h * (i + 0.5));

    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n)
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
        for (int i = lb; i <= ub; ++i) {
            sumloc += func(a + h * (i + 0.5));
        }

        #pragma omp atomic
        sum += sumloc;
    }

    sum *= h;
    return sum;
}

const double PI = 3.14159265358979323846;
const double A = -4.0;
const double B = 4.0;
const int NSTEPS = 40000000;

double run_serial()
{
    double t0 = timeChrono();
    double res = integrate(A, B, NSTEPS);
    double t1 = timeChrono();
    double t = t1 - t0;
    std::printf("Result (serial):   %.12f; error %.12f\n",
                res, std::fabs(res - std::sqrt(PI)));
    return t;
}

double run_parallel()
{
    double t0 = timeChrono();
    double res = integrate_omp(func, A, B, NSTEPS);
    double t1 = timeChrono();
    double t = t1 - t0;
    std::printf("Result (parallel): %.12f; error %.12f\n",
                res, std::fabs(res - std::sqrt(PI)));
    return t;
}

int main()
{
    std::printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n",
                A, B, NSTEPS);

    double tserial = run_serial();
    double tparallel = run_parallel();

    std::printf("Execution time (serial):   %.6f s\n", tserial);
    std::printf("Execution time (parallel): %.6f s\n", tparallel);
    std::printf("Speedup: %.2f\n", tserial / tparallel);

    return 0;
}

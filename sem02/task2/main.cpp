#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cmath>

double func(double x) {
    return std::exp(-x * x);
}

double integrate_omp(
    double (*f)(double),
    double a,
    double b,
    int nsteps = 40'000'000
) {
    double h = (b - a) / nsteps;
    double sum = 0.0;
    
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = nsteps / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (nsteps - 1) : (lb + items_per_thread - 1);
        
        double sumloc = 0.0;
        for (int i = lb; i <= ub; i++) {
            sumloc += f(a + h * (i + 0.5));
        }
        
        #pragma omp atomic
        sum += sumloc;
    }
    
    sum *= h;
    return sum;
}

double run_parallel() {
    const double a = -4.0;
    const double b = 4.0;
    const int nsteps = 40'000'000;

    const auto start = std::chrono::steady_clock::now();
    integrate_omp(func, a, b, nsteps);
    const auto end = std::chrono::steady_clock::now();
    
    const std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
}

int main() {
    const int tests_num = 100;
    std::vector<int> test_threads{1, 2, 4, 7, 8, 16, 20, 40};
    
    std::ofstream fout("integrate_time.csv");
    fout << "threads,time" << std::endl;

    for (int nthreads : test_threads) {
        omp_set_num_threads(nthreads);
        
        for (int t = 0; t < tests_num; ++t) {
            double time = run_parallel();
            fout << nthreads << "," << time << std::endl;
        }
    }

    fout.close();
    return 0;
}
#include <omp.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <chrono>
#include <vector>
#include <iomanip>

double timeChrono()
{
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> seconds = now.time_since_epoch();
    return seconds.count();
}

void init_system(std::size_t n,
                 std::unique_ptr<double[]> &a,
                 std::unique_ptr<double[]> &b)
{
    a = std::make_unique<double[]>(n * n);
    b = std::make_unique<double[]>(n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            a[i * n + j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = static_cast<double>(n) + 1.0;
    }
}

double solve_omp_static(const double *a, const double *b,
                        std::size_t n, double epsilon, std::size_t num_iter,
                        int chunk)
{
    std::unique_ptr<double[]> x_old = std::make_unique<double[]>(n);
    std::unique_ptr<double[]> x_new = std::make_unique<double[]>(n);
    for (std::size_t i = 0; i < n; ++i) x_old[i] = 0.0;

    double t0 = timeChrono();

#pragma omp parallel
    {
        for (std::size_t iter = 0; iter < num_iter; ++iter)
        {
#pragma omp for schedule(static, chunk)
            for (std::size_t i = 0; i < n; ++i) {
                double sum = 0.0;
                std::size_t curr = i * n;
                for (std::size_t j = 0; j < n; ++j)
                    if (i != j) sum += a[curr + j] * x_old[j];
                x_new[i] = (b[i] - sum) / a[curr + i];
            }

            double max_diff = 0.0;
#pragma omp for reduction(max:max_diff) schedule(static, chunk)
            for (std::size_t i = 0; i < n; ++i) {
                double d = std::fabs(x_new[i] - x_old[i]);
                if (d > max_diff) max_diff = d;
            }

            bool stop = false;
#pragma omp single
            {
                if (max_diff < epsilon)
                    stop = true;
            }

            if (stop)
                break;

#pragma omp for schedule(static, chunk)
            for (std::size_t i = 0; i < n; ++i)
                x_old[i] = x_new[i];
        }
    }

    double t1 = timeChrono();
    return t1 - t0;
}

double solve_serial(const double *a, const double *b,
                    std::size_t n, double epsilon, std::size_t num_iter)
{
    std::unique_ptr<double[]> x_old = std::make_unique<double[]>(n);
    std::unique_ptr<double[]> x_new = std::make_unique<double[]>(n);
    for (std::size_t i = 0; i < n; ++i) x_old[i] = 0.0;

    double t0 = timeChrono();

    for (std::size_t iter = 0; iter < num_iter; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            std::size_t curr = i * n;
            for (std::size_t j = 0; j < n; ++j)
                if (i != j) sum += a[curr + j] * x_old[j];
            x_new[i] = (b[i] - sum) / a[curr + i];
        }
        double max_diff = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double d = std::fabs(x_new[i] - x_old[i]);
            if (d > max_diff) max_diff = d;
        }
        if (max_diff < epsilon)
            break;
        for (std::size_t i = 0; i < n; ++i)
            x_old[i] = x_new[i];
    }

    double t1 = timeChrono();
    return t1 - t0;
}

int main(int argc, char* argv[])
{
    std::size_t n = 3000;
    if (argc > 1)
        n = static_cast<std::size_t>(std::atoll(argv[1]));

    double epsilon       = 1e-6;
    std::size_t num_iter = 1000;

    std::cout << "n = " << n
              << ", epsilon = " << epsilon
              << ", num_iter = " << num_iter << "\n";

    std::unique_ptr<double[]> A, B;
    init_system(n, A, B);

    double T_serial = solve_serial(A.get(), B.get(), n, epsilon, num_iter);
    std::cout << "T_serial = " << T_serial << " s\n\n";

    int threads_arr[] = {1, 2, 4, 8, 16, 20, 40};
    int nthreads = sizeof(threads_arr)/sizeof(threads_arr[0]);

    std::vector<int> threads(threads_arr, threads_arr + nthreads);

    std::vector<double> Sp_static100, Sp_static200, Sp_static_nt;

    std::cout << std::fixed << std::setprecision(4);

    for (int k = 0; k < nthreads; ++k) {
        int p = threads_arr[k];
        omp_set_num_threads(p);

        int chunk100 = 100;
        int chunk200 = 200;
        int chunkNT  = std::max(1, (int)(n / p));

        double t_s100 = solve_omp_static(A.get(), B.get(), n, epsilon, num_iter,
                                         chunk100);
        double t_s200 = solve_omp_static(A.get(), B.get(), n, epsilon, num_iter,
                                         chunk200);
        double t_snt  = solve_omp_static(A.get(), B.get(), n, epsilon, num_iter,
                                         chunkNT);

        double S_s100 = T_serial / t_s100;
        double S_s200 = T_serial / t_s200;
        double S_snt  = T_serial / t_snt;

        Sp_static100.push_back(S_s100);
        Sp_static200.push_back(S_s200);
        Sp_static_nt.push_back(S_snt);

        std::cout << "threads=" << p
                  << "  S_stat(100)=" << S_s100
                  << "  S_stat(200)=" << S_s200
                  << "  S_stat(n/p)=" << S_snt
                  << "\n";
    }

    std::cout << std::fixed << std::setprecision(2);

    auto print_vec = [](const char* name, const std::vector<double>& v){
        std::cout << name << " = [";
        for (std::size_t i = 0; i < v.size(); ++i) {
            std::cout << v[i];
            if (i + 1 < v.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    };

    std::cout << "\nthreads = [";
    for (std::size_t i = 0; i < threads.size(); ++i) {
        std::cout << threads[i];
        if (i + 1 < threads.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    print_vec("Sp_static100", Sp_static100);
    print_vec("Sp_static200", Sp_static200);
    print_vec("Sp_static_nt", Sp_static_nt);

    return 0;
}

// нужно добавить вывод времени для таблицы 
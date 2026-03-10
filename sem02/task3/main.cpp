#include <omp.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <chrono>


double timeChrono()
{
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::duration<double> seconds = now.time_since_epoch();
    return seconds.count();
}

// инициализация матрицы A и векторов b, x_old, x_new
// A_ii = 2.0, A_ij = 1.0 при i != j
// b_i = N + 1, x_old = 0
void init_system(std::size_t n,
                 std::unique_ptr<double[]> &a,
                 std::unique_ptr<double[]> &b,
                 std::unique_ptr<double[]> &x_old,
                 std::unique_ptr<double[]> &x_new)
{
    a     = std::make_unique<double[]>(n * n);
    b     = std::make_unique<double[]>(n);
    x_old = std::make_unique<double[]>(n);
    x_new = std::make_unique<double[]>(n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            a[i * n + j] = (i == j) ? 2.0 : 1.0;
        }
        x_old[i] = 0.0;
        b[i]     = static_cast<double>(n) + 1.0;
    }
}

// Вариант 1: для каждого цикла своя параллельная секция #pragma omp parallel for
// Итерационный метод (Якоби), критерий ||x_new - x_old||_inf < epsilon
double solve_omp_v1(std::size_t n, double epsilon, std::size_t num_iter)
{
    std::unique_ptr<double[]> a, b, x_old, x_new;
    init_system(n, a, b, x_old, x_new);

    double t0 = timeChrono();

    for (std::size_t iter = 0; iter < num_iter; ++iter) {

        // вычисление нового x
        #pragma omp parallel for schedule(guided)
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            std::size_t curr = i * n;
            for (std::size_t j = 0; j < n; ++j) {
                if (i != j) sum += a[curr + j] * x_old[j];
            }
            x_new[i] = (b[i] - sum) / a[curr + i];
        }

        double max_diff = 0.0;
        // максимум разности ||x_new - x_old||_inf
        #pragma omp parallel for reduction(max:max_diff) schedule(guided)
        for (std::size_t i = 0; i < n; ++i) {
            double curr_diff = std::fabs(x_new[i] - x_old[i]);
            if (curr_diff > max_diff) max_diff = curr_diff;
        }

        if (max_diff < epsilon)
            break;

        // копирование x_new -> x_old
        #pragma omp parallel for schedule(guided)
        for (std::size_t i = 0; i < n; ++i)
            x_old[i] = x_new[i];
    }

    double t1 = timeChrono();
    return t1 - t0;
}

// Вариант 2: одна параллельная секция #pragma omp parallel,
// внутри итерационный цикл, распределение циклов через #pragma omp for
double solve_omp_v2(std::size_t n, double epsilon, std::size_t num_iter)
{
    std::unique_ptr<double[]> a, b, x_old, x_new;
    init_system(n, a, b, x_old, x_new);

    double t0 = timeChrono();

    #pragma omp parallel
    {
        for (std::size_t iter = 0; iter < num_iter; ++iter) {

            // вычисление нового x
            #pragma omp for schedule(guided)
            for (std::size_t i = 0; i < n; ++i) {
                double sum = 0.0;
                std::size_t curr = i * n;
                for (std::size_t j = 0; j < n; ++j) {
                    if (i != j) sum += a[curr + j] * x_old[j];
                }
                x_new[i] = (b[i] - sum) / a[curr + i];
            }

            double max_diff = 0.0;
            // максимум разности ||x_new - x_old||_inf
            #pragma omp for reduction(max:max_diff) schedule(guided)
            for (std::size_t i = 0; i < n; ++i) {
                double curr_diff = std::fabs(x_new[i] - x_old[i]);
                if (curr_diff > max_diff) max_diff = curr_diff;
            }

            bool stop = false;
            // один поток проверяет критерий останова
            #pragma omp single
            {
                if (max_diff < epsilon)
                    stop = true;
            }

            if (stop)
                break;

            // копирование x_new -> x_old
            #pragma omp for schedule(guided)
            for (std::size_t i = 0; i < n; ++i)
                x_old[i] = x_new[i];
        }
    }

    double t1 = timeChrono();
    return t1 - t0;
}

int main(int argc, char* argv[])
{
    std::size_t n = 1500;
    if (argc > 1)
        n = static_cast<std::size_t>(std::atoll(argv[1]));

    double epsilon       = 1e-6;
    std::size_t num_iter = 1000;

    std::cout << "n = " << n
              << ", epsilon = " << epsilon
              << ", num_iter = " << num_iter << "\n";

    double t_v1 = solve_omp_v1(n, epsilon, num_iter);
    double t_v2 = solve_omp_v2(n, epsilon, num_iter);

    std::cout << "Time OMP v1 (many parallel for): " << t_v1 << " s\n";
    std::cout << "Time OMP v2 (single parallel):   " << t_v2 << " s\n";

    return 0;
}

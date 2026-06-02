#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace po = boost::program_options;

static constexpr double kCornerBL = 10.0;
static constexpr double kCornerBR = 20.0;
static constexpr double kCornerTR = 30.0;
static constexpr double kCornerTL = 20.0;
static constexpr int    kMaxIterations = 1'100'000;

// индексация 2D -> 1D
inline std::size_t idx(int j, int i, int m) noexcept {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(m)
         + static_cast<std::size_t>(i);
}

// граничные условия
void set_boundary(double* grid, int m, int n) {
    if (m <= 0 || n <= 0) return;

    grid[idx(0,     0,   m)] = kCornerBL;
    grid[idx(0,     m-1, m)] = kCornerBR;
    grid[idx(n-1,   m-1, m)] = kCornerTR;
    grid[idx(n-1,   0,   m)] = kCornerTL;

    if (m > 2) {
        const double denom = static_cast<double>(m - 1);
        for (int i = 1; i < m - 1; ++i) {
            const double t = static_cast<double>(i) / denom;
            grid[idx(0,   i, m)] = kCornerBL + (kCornerBR - kCornerBL) * t;
            grid[idx(n-1, i, m)] = kCornerTL + (kCornerTR - kCornerTL) * t;
        }
    }
    if (n > 2) {
        const double denom = static_cast<double>(n - 1);
        for (int j = 1; j < n - 1; ++j) {
            const double t = static_cast<double>(j) / denom;
            grid[idx(j, 0,   m)] = kCornerBL + (kCornerTL - kCornerBL) * t;
            grid[idx(j, m-1, m)] = kCornerBR + (kCornerTR - kCornerBR) * t;
        }
    }
}

// обнуление + границы - версия для сырых указателей (работает с .get())
void initialize(double* a, double* b, int m, int n) {
    const std::size_t count =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    std::memset(a, 0, count * sizeof(double));
    std::memset(b, 0, count * sizeof(double));
    set_boundary(a, m, n);
    set_boundary(b, m, n);
}

void print_grid(const double* grid, int m, int n) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::printf("%10.6f", grid[idx(j, i, m)]);
            if (i + 1 < m) std::printf(" ");
        }
        std::printf("\n");
    }
}

inline double wtime() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Params {
    int    size     = 128;
    double tol      = 1.0e-6;
    int    max_iter = kMaxIterations;
    int    cores    = 0;      // для multicore
};

Params parse_args(int argc, char** argv) {
    Params p;
    po::options_description desc("heat_all (cpu_onecore/cpu_multicore/gpu_base)");
    desc.add_options()
        ("help,h", "print help")
        ("size,s", po::value<int>(&p.size)->default_value(p.size), "grid size NxN")
        ("eps,e",  po::value<double>(&p.tol)->default_value(p.tol), "tolerance")
        ("tol,t",  po::value<double>(&p.tol), "alias for --eps")
        ("max-iters,m", po::value<int>(&p.max_iter)->default_value(p.max_iter),
                        "max iterations")
        ("cores,c", po::value<int>(&p.cores), "CPU cores for multicore");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        std::exit(0);
    }
    if (p.size < 3) {
        std::cerr << "size must be >= 3\n";
        std::exit(1);
    }
    if (p.max_iter < 1) {
        std::cerr << "max-iters must be >= 1\n";
        std::exit(1);
    }
    if (p.max_iter > kMaxIterations) {
        std::cerr << "max-iters capped at " << kMaxIterations << "\n";
        p.max_iter = kMaxIterations;
    }
    return p;
}

void setup_cores(int cores) {
    if (cores <= 0) return;
    std::string s = std::to_string(cores);
    setenv("ACC_NUM_CORES",   s.c_str(), 1);
    setenv("OMP_NUM_THREADS", s.c_str(), 1);
    setenv("OMP_PROC_BIND",   "close",   0);
    setenv("OMP_PLACES",      "cores",   0);
}

int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);
    const int m = p.size, n = p.size;
    const std::size_t count = m * n;

    setup_cores(p.cores);

    // Используем умные указатели - НЕТ delete[]
    std::unique_ptr<double[]> buf_a = std::make_unique<double[]>(count);
    std::unique_ptr<double[]> buf_b = std::make_unique<double[]>(count);
    
    // Передаём сырые указатели через .get()
    initialize(buf_a.get(), buf_b.get(), m, n);

    double error = 1.0;
    int iter = 0;

    const double t0 = wtime();

    // Получаем сырые указатели для OpenACC
    double* raw_a = buf_a.get();
    double* raw_b = buf_b.get();

    // ИСПРАВЛЕННАЯ СЕКЦИЯ ДЛЯ GPU
    #pragma acc data copyin(raw_a[0:count]) create(raw_b[0:count])
    {
        for (iter = 0; iter < p.max_iter; ++iter) {
            error = 0.0;

            // A -> B + вычисление ошибки
            #pragma acc parallel loop collapse(2) reduction(max:error)
            for (int j = 1; j < n - 1; ++j) {
                for (int i = 1; i < m - 1; ++i) {
                    const std::size_t id = idx(j, i, m);
                    const double v = 0.25 * (raw_a[idx(j, i+1, m)] +
                                             raw_a[idx(j, i-1, m)] +
                                             raw_a[idx(j-1, i, m)] +
                                             raw_a[idx(j+1, i, m)]);
                    raw_b[id] = v;
                    double diff = std::fabs(v - raw_a[id]);
                    if (diff > error) error = diff;
                }
            }

            // B -> A копированием
            #pragma acc parallel loop collapse(2)
            for (int j = 1; j < n - 1; ++j) {
                for (int i = 1; i < m - 1; ++i) {
                    const std::size_t id = idx(j, i, m);
                    raw_a[id] = raw_b[id];
                }
            }

            if (error <= p.tol) { ++iter; break; }
        }
    }

    double t1 = wtime();

    std::cout << std::fixed << std::setprecision(6)
              << "time:       " << (t1 - t0) << " s\n"
              << "iterations: " << iter      << "\n"
              << "error:      " << std::scientific << error << "\n";

    if (p.size == 10 || p.size == 13) print_grid(raw_a, m, n);

    // НЕТ delete[] - память освободится автоматически
    return 0;
}
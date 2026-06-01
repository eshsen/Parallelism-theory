#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace po = boost::program_options;

static constexpr double kCornerBL = 10.0;
static constexpr double kCornerBR = 20.0;
static constexpr double kCornerTR = 30.0;
static constexpr double kCornerTL = 20.0;
static constexpr int    kMaxIterations     = 1'100'000;
static constexpr int    kDefaultCheckEvery = 10000;

#ifndef VEC_LEN
#define VEC_LEN 128
#endif

inline std::size_t idx(int j, int i, int m) noexcept {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(m)
         + static_cast<std::size_t>(i);
}

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
    int    check    = kDefaultCheckEvery;
};

Params parse_args(int argc, char** argv) {
    Params p;
    po::options_description desc("gpu3");
    desc.add_options()
        ("help,h", "print help")
        ("size,s", po::value<int>(&p.size)->default_value(p.size), "grid size NxN")
        ("eps,e",  po::value<double>(&p.tol)->default_value(p.tol), "tolerance")
        ("tol,t",  po::value<double>(&p.tol), "alias for --eps")
        ("max-iters,m", po::value<int>(&p.max_iter)->default_value(p.max_iter),
                        "max iterations")
        ("check-interval,k", po::value<int>(&p.check)->default_value(p.check),
                             "error check period");

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
    if (p.size == 10 || p.size == 13) p.check = 1;
    if (p.check < 1) p.check = 1;
    return p;
}

int main(int argc, char** argv) {
    Params p = parse_args(argc, argv);
    const int m = p.size, n = p.size;
    const std::size_t count =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

    double* buf_a = new double[count];
    double* buf_b = new double[count];
    initialize(buf_a, buf_b, m, n);

    double error = 1.0;
    int    iter  = 0;
    bool   cur_a = true;

    const double t0 = wtime();

    #pragma acc enter data copyin(buf_a[0:count], buf_b[0:count])

    while (iter < p.max_iter && error > p.tol) {
        // асинхронный пакет шагов
        for (int s = 0; s < p.check && iter < p.max_iter; ++s, ++iter) {
            if (cur_a) {
                #pragma acc parallel loop collapse(2) vector_length(VEC_LEN) \
                        async(1) present(buf_a[0:count], buf_b[0:count])
                for (int j = 1; j < n - 1; ++j) {
                    for (int i = 1; i < m - 1; ++i) {
                        const std::size_t id = idx(j, i, m);
                        buf_b[id] = 0.25 * (buf_a[idx(j, i+1, m)] +
                                            buf_a[idx(j, i-1, m)] +
                                            buf_a[idx(j-1, i, m)] +
                                            buf_a[idx(j+1, i, m)]);
                    }
                }
            } else {
                #pragma acc parallel loop collapse(2) vector_length(VEC_LEN) \
                        async(1) present(buf_a[0:count], buf_b[0:count])
                for (int j = 1; j < n - 1; ++j) {
                    for (int i = 1; i < m - 1; ++i) {
                        const std::size_t id = idx(j, i, m);
                        buf_a[id] = 0.25 * (buf_b[idx(j, i+1, m)] +
                                            buf_b[idx(j, i-1, m)] +
                                            buf_b[idx(j-1, i, m)] +
                                            buf_b[idx(j+1, i, m)]);
                    }
                }
            }
            cur_a = !cur_a;
        }

        #pragma acc wait(1)
        error = 0.0;

        if (cur_a) {
            #pragma acc parallel loop collapse(2) vector_length(VEC_LEN) \
                    present(buf_a[0:count], buf_b[0:count]) reduction(max:error)
            for (int j = 1; j < n - 1; ++j) {
                for (int i = 1; i < m - 1; ++i) {
                    const std::size_t id = idx(j, i, m);
                    double diff = std::fabs(buf_a[id] - buf_b[id]);
                    if (diff > error) error = diff;
                }
            }
        } else {
            #pragma acc parallel loop collapse(2) vector_length(VEC_LEN) \
                    present(buf_a[0:count], buf_b[0:count]) reduction(max:error)
            for (int j = 1; j < n - 1; ++j) {
                for (int i = 1; i < m - 1; ++i) {
                    const std::size_t id = idx(j, i, m);
                    double diff = std::fabs(buf_b[id] - buf_a[id]);
                    if (diff > error) error = diff;
                }
            }
        }
    }

    double* solution = cur_a ? buf_a : buf_b;

    if (p.size == 10 || p.size == 13) {
        #pragma acc update host(solution[0:count])
        print_grid(solution, m, n);
    } else {
        #pragma acc update host(solution[0:count])
    }

    #pragma acc exit data delete(buf_a[0:count], buf_b[0:count])

    std::cout << std::fixed << std::setprecision(6)
            << "time:       " << (wtime() - t0) << " s\n"
            << "iterations: " << iter      << "\n"
            << "error:      " << std::scientific << error << "\n";

    delete[] buf_a;
    delete[] buf_b;
    return 0;
}
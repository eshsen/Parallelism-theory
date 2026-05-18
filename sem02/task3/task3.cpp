#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>

struct Config {
    int n = 10000;
    int max_iter = 20000;
    double tol = 1e-8;
    double tau = 0.02;
    int variant = 1;
    int threads = 1;
    omp_sched_t schedule_kind = omp_sched_static;
    int chunk_size = 0;
};

struct Result {
    int iters = 0;
    double residual = 0.0;
    double seconds = 0.0;
};

struct VariantRow {
    int variant = 0;
    int threads = 0;
    int n = 0;
    int max_iter = 0;
    double tau = 0.0;
    double tol = 0.0;
    int iterations = 0;
    double residual = 0.0;
    double time_sec = 0.0;
    std::string schedule;
    int chunk = 0;
    double speedup = 0.0;
    double efficiency = 0.0;
};

struct SchedulePlan {
    omp_sched_t kind = omp_sched_static;
    const char *name = "static";
    int chunk = 0;
};

static inline double a_ij(int i, int j) {
    return (i == j) ? 4.0 : (1.0 / (1.0 + std::abs(i - j)));
}

static void build_problem(int n, std::vector<double> &b) {
    // Deterministic fill: always the same matrix/vector for any thread count.
    const std::vector<double> x_true(n, 1.0);
    b.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += a_ij(i, j) * x_true[j];
        }
        b[i] = sum;
    }
}

static Result richardson_variant1(const Config &cfg, const std::vector<double> &b) {
    const int n = cfg.n;
    std::vector<double> x(n, 0.0), r(n, 0.0), ax(n, 0.0), x_new(n, 0.0);

    double residual = std::numeric_limits<double>::infinity();
    int iter = 0;
    const double t0 = omp_get_wtime();

    for (; iter < cfg.max_iter && residual > cfg.tol; ++iter) {
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += a_ij(i, j) * x[j];
            }
            ax[i] = sum;
        }

        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - ax[i];
        }

        double norm_sq = 0.0;
        #pragma omp parallel for schedule(runtime) reduction(+ : norm_sq)
        for (int i = 0; i < n; ++i) {
            norm_sq += r[i] * r[i];
        }
        residual = std::sqrt(norm_sq);

        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < n; ++i) {
            x_new[i] = x[i] + cfg.tau * r[i];
        }

        x.swap(x_new);
    }

    const double t1 = omp_get_wtime();
    return {iter, residual, t1 - t0};
}

static Result richardson_variant2(const Config &cfg, const std::vector<double> &b) {
    const int n = cfg.n;
    std::vector<double> x(n, 0.0), r(n, 0.0), ax(n, 0.0), x_new(n, 0.0);

    double residual = std::numeric_limits<double>::infinity();
    double norm_sq = 0.0;
    int iter = 0;
    const double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(x, r, ax, x_new, b, residual, norm_sq, iter, cfg, n)
    {
        for (int k = 0; k < cfg.max_iter; ++k) {
            #pragma omp for schedule(runtime)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += a_ij(i, j) * x[j];
                }
                ax[i] = sum;
            }

            #pragma omp for schedule(runtime)
            for (int i = 0; i < n; ++i) {
                r[i] = b[i] - ax[i];
            }

            #pragma omp single
            {
                norm_sq = 0.0;
            }

            #pragma omp for schedule(runtime) reduction(+ : norm_sq)
            for (int i = 0; i < n; ++i) {
                norm_sq += r[i] * r[i];
            }

            #pragma omp single
            {
                residual = std::sqrt(norm_sq);
                iter = k + 1;
            }

            #pragma omp for schedule(runtime)
            for (int i = 0; i < n; ++i) {
                x_new[i] = x[i] + cfg.tau * r[i];
            }

            #pragma omp for schedule(runtime)
            for (int i = 0; i < n; ++i) {
                x[i] = x_new[i];
            }

            #pragma omp barrier
            if (residual <= cfg.tol) {
                break;
            }
        }
    }

    const double t1 = omp_get_wtime();
    return {iter, residual, t1 - t0};
}

static const char *schedule_name(omp_sched_t kind) {
    if (kind == omp_sched_static) {
        return "static";
    }
    if (kind == omp_sched_dynamic) {
        return "dynamic";
    }
    if (kind == omp_sched_guided) {
        return "guided";
    }
    if (kind == omp_sched_auto) {
        return "auto";
    }
    return "unknown";
}

static Result run_single_case(const Config &cfg, const std::vector<double> &b) {
    omp_set_num_threads(cfg.threads);
    omp_set_schedule(cfg.schedule_kind, cfg.chunk_size);
    return (cfg.variant == 1) ? richardson_variant1(cfg, b) : richardson_variant2(cfg, b);
}

static void write_variants_csv(const std::string &path, const std::vector<VariantRow> &rows) {
    std::ofstream out(path);
    out << "variant,threads,n,max_iter,tau,tol,iterations,residual,time_sec,schedule,chunk,speedup,efficiency\n";
    out << std::fixed << std::setprecision(6);
    for (const VariantRow &row : rows) {
        out << row.variant << ','
            << row.threads << ','
            << row.n << ','
            << row.max_iter << ','
            << row.tau << ','
            << row.tol << ','
            << row.iterations << ','
            << row.residual << ','
            << row.time_sec << ','
            << row.schedule << ','
            << row.chunk << ','
            << row.speedup << ','
            << row.efficiency << '\n';
    }
}

static void write_schedule_csv(const std::string &path, const std::vector<VariantRow> &rows) {
    std::ofstream out(path);
    out << "variant,threads,n,max_iter,tau,tol,iterations,residual,time_sec,schedule,chunk\n";
    out << std::fixed << std::setprecision(6);
    for (const VariantRow &row : rows) {
        out << row.variant << ','
            << row.threads << ','
            << row.n << ','
            << row.max_iter << ','
            << row.tau << ','
            << row.tol << ','
            << row.iterations << ','
            << row.residual << ','
            << row.time_sec << ','
            << row.schedule << ','
            << row.chunk << '\n';
    }
}

int main() {
    // All experiment parameters are configured here.
    const int problem_n = 10000;
    const int problem_max_iter = 20000;
    const double problem_tau = 0.02;
    const double problem_tol = 1e-8;
    const omp_sched_t baseline_schedule = omp_sched_static;
    const int baseline_chunk = 0;

    const int max_threads = omp_get_max_threads();
    const int schedule_variant = 2;
    const int schedule_threads = (max_threads >= 8) ? 8 : max_threads;

    const std::vector<SchedulePlan> schedule_plans = {
        {omp_sched_static, "static", 0},
        {omp_sched_static, "static", 1},
        {omp_sched_static, "static", 16},
        {omp_sched_dynamic, "dynamic", 1},
        {omp_sched_dynamic, "dynamic", 16},
        {omp_sched_guided, "guided", 1},
        {omp_sched_guided, "guided", 16},
    };

    Config cfg;
    cfg.n = problem_n;
    cfg.max_iter = problem_max_iter;
    cfg.tau = problem_tau;
    cfg.tol = problem_tol;
    cfg.schedule_kind = baseline_schedule;
    cfg.chunk_size = baseline_chunk;

    std::vector<double> b;
    build_problem(cfg.n, b);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Problem: n=" << cfg.n
              << " max_iter=" << cfg.max_iter
              << " tau=" << cfg.tau
              << " tol=" << cfg.tol
              << " threads_range=1.." << max_threads << '\n';

    std::vector<VariantRow> variant_rows;
    for (int variant = 1; variant <= 2; ++variant) {
        double base_time = 0.0;
        cfg.variant = variant;
        for (int threads = 1; threads <= max_threads; ++threads) {
            cfg.threads = threads;
            const Result res = run_single_case(cfg, b);
            if (threads == 1) {
                base_time = res.seconds;
            }
            VariantRow row;
            row.variant = variant;
            row.threads = threads;
            row.n = cfg.n;
            row.max_iter = cfg.max_iter;
            row.tau = cfg.tau;
            row.tol = cfg.tol;
            row.iterations = res.iters;
            row.residual = res.residual;
            row.time_sec = res.seconds;
            row.schedule = schedule_name(cfg.schedule_kind);
            row.chunk = cfg.chunk_size;
            row.speedup = base_time / res.seconds;
            row.efficiency = row.speedup / static_cast<double>(threads);
            variant_rows.push_back(row);
            std::cout << "variant=" << row.variant
                      << " threads=" << row.threads
                      << " time_sec=" << row.time_sec
                      << " speedup=" << row.speedup
                      << " efficiency=" << row.efficiency << '\n';
        }
    }
    write_variants_csv("task3_variants.csv", variant_rows);
    std::cout << "Wrote task3_variants.csv\n";

    std::vector<VariantRow> schedule_rows;
    cfg.variant = schedule_variant;
    cfg.threads = schedule_threads;
    for (const SchedulePlan &plan : schedule_plans) {
        cfg.schedule_kind = plan.kind;
        cfg.chunk_size = plan.chunk;
        const Result res = run_single_case(cfg, b);
        VariantRow row;
        row.variant = cfg.variant;
        row.threads = cfg.threads;
        row.n = cfg.n;
        row.max_iter = cfg.max_iter;
        row.tau = cfg.tau;
        row.tol = cfg.tol;
        row.iterations = res.iters;
        row.residual = res.residual;
        row.time_sec = res.seconds;
        row.schedule = plan.name;
        row.chunk = plan.chunk;
        schedule_rows.push_back(row);
        std::cout << "schedule_test variant=" << row.variant
                  << " threads=" << row.threads
                  << " schedule=" << row.schedule
                  << " chunk=" << row.chunk
                  << " time_sec=" << row.time_sec << '\n';
    }
    write_schedule_csv("task3_schedule.csv", schedule_rows);
    std::cout << "Wrote task3_schedule.csv\n";

    return 0;
}
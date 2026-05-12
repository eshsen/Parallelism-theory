#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

namespace
{
constexpr int kDefaultRepeats = 50;
const std::vector<int> kThreadCounts = {1, 2, 4, 7, 8, 16, 20, 40};

struct SolverResult
{
  double elapsed_sec = 0.0;
  int iterations = 0;
  double diff_norm = 0.0;
  double error_norm = 0.0;
};

struct Workspace
{
  std::vector<double> x;
  std::vector<double> x_new;

  explicit Workspace(int n)
      : x(static_cast<std::size_t>(n)),
        x_new(static_cast<std::size_t>(n))
  {
  }

  void reset(int n)
  {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
      x[static_cast<std::size_t>(i)] = 0.0;
      x_new[static_cast<std::size_t>(i)] = 0.0;
    }
  }
};

struct ScheduleConfig
{
  omp_sched_t kind;
  int chunk;
  const char *name;
};

void print_system_info()
{
  std::cout << "\n=== System information ===\n";
  std::system("lscpu | grep 'Model name'");
  std::system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'N/A'");
  std::system("numactl --hardware 2>/dev/null | grep -E 'available|node [0-9]+ size' || echo 'NUMA info not available'");
  std::system("cat /etc/os-release 2>/dev/null | grep 'PRETTY_NAME' | cut -d'=' -f2 | tr -d '\"'");
  std::cout << "=====================================\n\n";
}

double compute_error_norm(const std::vector<double> &x)
{
  double err = 0.0;
#pragma omp parallel for reduction(+ : err) schedule(static)
  for (int i = 0; i < static_cast<int>(x.size()); ++i)
  {
    const double d = x[static_cast<std::size_t>(i)] - 1.0;
    err += d * d;
  }
  return std::sqrt(err);
}

SolverResult solve_variant1(Workspace &ws, int n, int max_iters, double eps, int threads, double tau_factor)
{
  ws.reset(n);
  omp_set_num_threads(threads);

  const double b_value = static_cast<double>(n + 1);
  const double tau = tau_factor / b_value;
  double diff = std::numeric_limits<double>::infinity();
  int iteration = 0;
  const double start = omp_get_wtime();

  while (iteration < max_iters && diff > eps)
  {
    double sum_x = 0.0;
#pragma omp parallel for reduction(+ : sum_x) schedule(static)
    for (int i = 0; i < n; ++i)
    {
      sum_x += ws.x[static_cast<std::size_t>(i)];
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
      const double ax = sum_x + ws.x[static_cast<std::size_t>(i)];
      ws.x_new[static_cast<std::size_t>(i)] = ws.x[static_cast<std::size_t>(i)] - tau * (ax - b_value);
    }

    diff = 0.0;
#pragma omp parallel for reduction(+ : diff) schedule(static)
    for (int i = 0; i < n; ++i)
    {
      const double d = ws.x_new[static_cast<std::size_t>(i)] - ws.x[static_cast<std::size_t>(i)];
      diff += d * d;
    }
    diff = std::sqrt(diff);
    std::swap(ws.x, ws.x_new);
    ++iteration;
  }

  return {omp_get_wtime() - start, iteration, diff, compute_error_norm(ws.x)};
}

SolverResult solve_variant2(Workspace &ws,
                            int n,
                            int max_iters,
                            double eps,
                            int threads,
                            double tau_factor,
                            bool runtime_schedule)
{
  ws.reset(n);
  omp_set_num_threads(threads);

  const double b_value = static_cast<double>(n + 1);
  const double tau = tau_factor / b_value;
  double diff = std::numeric_limits<double>::infinity();
  double sum_x = 0.0;
  double diff_sq = 0.0;
  int iteration = 0;
  bool stop = false;
  const double start = omp_get_wtime();

#pragma omp parallel shared(ws, diff, sum_x, diff_sq, iteration, stop)
  {
    while (true)
    {
#pragma omp single
      {
        stop = (iteration >= max_iters || diff <= eps);
        sum_x = 0.0;
        diff_sq = 0.0;
      }
#pragma omp barrier
      if (stop)
      {
        break;
      }

      double local_sum = 0.0;
      if (runtime_schedule)
      {
#pragma omp for schedule(runtime) nowait
        for (int i = 0; i < n; ++i)
        {
          local_sum += ws.x[static_cast<std::size_t>(i)];
        }
      }
      else
      {
#pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i)
        {
          local_sum += ws.x[static_cast<std::size_t>(i)];
        }
      }
#pragma omp atomic update
      sum_x += local_sum;
#pragma omp barrier

      if (runtime_schedule)
      {
#pragma omp for schedule(runtime)
        for (int i = 0; i < n; ++i)
        {
          const double ax = sum_x + ws.x[static_cast<std::size_t>(i)];
          ws.x_new[static_cast<std::size_t>(i)] = ws.x[static_cast<std::size_t>(i)] - tau * (ax - b_value);
        }
      }
      else
      {
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i)
        {
          const double ax = sum_x + ws.x[static_cast<std::size_t>(i)];
          ws.x_new[static_cast<std::size_t>(i)] = ws.x[static_cast<std::size_t>(i)] - tau * (ax - b_value);
        }
      }

      double local_diff = 0.0;
      if (runtime_schedule)
      {
#pragma omp for schedule(runtime) nowait
        for (int i = 0; i < n; ++i)
        {
          const double d = ws.x_new[static_cast<std::size_t>(i)] - ws.x[static_cast<std::size_t>(i)];
          local_diff += d * d;
        }
      }
      else
      {
#pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i)
        {
          const double d = ws.x_new[static_cast<std::size_t>(i)] - ws.x[static_cast<std::size_t>(i)];
          local_diff += d * d;
        }
      }
#pragma omp atomic update
      diff_sq += local_diff;
#pragma omp barrier

#pragma omp single
      {
        diff = std::sqrt(diff_sq);
        std::swap(ws.x, ws.x_new);
        ++iteration;
      }
#pragma omp barrier
    }
  }

  return {omp_get_wtime() - start, iteration, diff, compute_error_norm(ws.x)};
}

double average_time(const std::vector<SolverResult> &results)
{
  double total = 0.0;
  for (const SolverResult &result : results)
  {
    total += result.elapsed_sec;
  }
  return total / static_cast<double>(results.size());
}
} // namespace

int main(int argc, char **argv)
{
  const int n = (argc > 1) ? std::stoi(argv[1]) : 20000;
  const int max_iters = (argc > 2) ? std::stoi(argv[2]) : 5000;
  const double eps = (argc > 3) ? std::stod(argv[3]) : 1e-6;
  const int repeats = (argc > 4) ? std::stoi(argv[4]) : kDefaultRepeats;
  const int schedule_threads = (argc > 5) ? std::stoi(argv[5]) : 8;
  const double tau_factor = (argc > 6) ? std::stod(argv[6]) : 1e-3;

  print_system_info();
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "N=" << n << ", max_iters=" << max_iters << ", eps=" << eps
            << ", repeats=" << repeats << ", tau_factor=" << tau_factor << "\n";

  Workspace workspace(n);

  std::ofstream runs_csv("iteration_runs.csv");
  std::ofstream summary_csv("iteration_summary.csv");
  if (!runs_csv || !summary_csv)
  {
    std::cerr << "Cannot open CSV files for writing\n";
    return 1;
  }

  runs_csv << "variant,threads,run,time_sec,iterations,diff_norm,error_norm\n";
  summary_csv << "variant,threads,avg_time_sec,speedup,efficiency,iterations,diff_norm,error_norm\n";

  double base_v1 = 0.0;
  double base_v2 = 0.0;
  for (const int threads : kThreadCounts)
  {
    std::vector<SolverResult> v1_results;
    std::vector<SolverResult> v2_results;
    v1_results.reserve(static_cast<std::size_t>(repeats));
    v2_results.reserve(static_cast<std::size_t>(repeats));

    for (int run = 1; run <= repeats; ++run)
    {
      const SolverResult v1 = solve_variant1(workspace, n, max_iters, eps, threads, tau_factor);
      v1_results.push_back(v1);
      runs_csv << "variant1," << threads << "," << run << "," << v1.elapsed_sec << ","
               << v1.iterations << "," << v1.diff_norm << "," << v1.error_norm << "\n";

      const SolverResult v2 = solve_variant2(workspace, n, max_iters, eps, threads, tau_factor, false);
      v2_results.push_back(v2);
      runs_csv << "variant2," << threads << "," << run << "," << v2.elapsed_sec << ","
               << v2.iterations << "," << v2.diff_norm << "," << v2.error_norm << "\n";
    }

    const double avg_v1 = average_time(v1_results);
    const double avg_v2 = average_time(v2_results);
    if (threads == 1)
    {
      base_v1 = avg_v1;
      base_v2 = avg_v2;
    }

    const SolverResult last_v1 = v1_results.back();
    const SolverResult last_v2 = v2_results.back();
    summary_csv << "variant1," << threads << "," << avg_v1 << ","
                << base_v1 / avg_v1 << "," << (base_v1 / avg_v1) / threads << ","
                << last_v1.iterations << "," << last_v1.diff_norm << "," << last_v1.error_norm << "\n";
    summary_csv << "variant2," << threads << "," << avg_v2 << ","
                << base_v2 / avg_v2 << "," << (base_v2 / avg_v2) / threads << ","
                << last_v2.iterations << "," << last_v2.diff_norm << "," << last_v2.error_norm << "\n";

    std::cout << "threads=" << std::setw(2) << threads
              << " v1_avg=" << avg_v1
              << " v2_avg=" << avg_v2 << "\n";
  }

  const int sched_threads = std::max(1, schedule_threads);
  const std::vector<ScheduleConfig> schedules = {
      {omp_sched_static, 1, "static,1"},
      {omp_sched_static, 64, "static,64"},
      {omp_sched_static, std::max(1, n / sched_threads), "static,N/T"},
      {omp_sched_dynamic, 1, "dynamic,1"},
      {omp_sched_dynamic, 64, "dynamic,64"},
      {omp_sched_dynamic, std::max(1, n / sched_threads), "dynamic,N/T"},
      {omp_sched_guided, 1, "guided,1"},
      {omp_sched_guided, 64, "guided,64"},
      {omp_sched_guided, std::max(1, n / sched_threads), "guided,N/T"}};

  std::ofstream schedule_runs_csv("iteration_schedule_runs.csv");
  std::ofstream schedule_summary_csv("iteration_schedule_summary.csv");
  if (!schedule_runs_csv || !schedule_summary_csv)
  {
    std::cerr << "Cannot open schedule CSV files for writing\n";
    return 1;
  }

  schedule_runs_csv << "schedule,threads,run,time_sec,iterations,diff_norm,error_norm\n";
  schedule_summary_csv << "schedule,threads,avg_time_sec,speedup,iterations,diff_norm,error_norm\n";

  double best_schedule_time = std::numeric_limits<double>::max();
  const char *best_schedule = "";
  for (const ScheduleConfig &cfg : schedules)
  {
    omp_set_schedule(cfg.kind, cfg.chunk);
    std::vector<SolverResult> results;
    results.reserve(static_cast<std::size_t>(repeats));

    for (int run = 1; run <= repeats; ++run)
    {
      const SolverResult result = solve_variant2(workspace, n, max_iters, eps, sched_threads, tau_factor, true);
      results.push_back(result);
      schedule_runs_csv << cfg.name << "," << sched_threads << "," << run << ","
                        << result.elapsed_sec << "," << result.iterations << ","
                        << result.diff_norm << "," << result.error_norm << "\n";
    }

    const double avg = average_time(results);
    const SolverResult last = results.back();
    schedule_summary_csv << cfg.name << "," << sched_threads << "," << avg << ","
                         << base_v2 / avg << "," << last.iterations << ","
                         << last.diff_norm << "," << last.error_norm << "\n";

    if (avg < best_schedule_time)
    {
      best_schedule_time = avg;
      best_schedule = cfg.name;
    }
  }

  std::cout << "\nBest schedule for variant2 with " << sched_threads
            << " threads: " << best_schedule << ", " << best_schedule_time << " sec\n";
  std::cout << "Saved: iteration_runs.csv, iteration_summary.csv, "
            << "iteration_schedule_runs.csv, iteration_schedule_summary.csv\n";
  return 0;
}
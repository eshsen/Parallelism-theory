#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <cstdlib>
#include <boost/program_options.hpp>
#ifdef _OPENACC
#include <openacc.h>
#endif
namespace po = boost::program_options;

#define EPS 0.001
#define MAX_ITER 1000000
#define SIZE 10
#define IND(i, j) ((i) * nx + (j))

static bool write_matrix_text(const std::string& path, const double* data, int n)
{
    std::ofstream out(path);
    if (!out) return false;
    out << n << '\n';
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out << data[i * n + j];
            if (j + 1 < n) out << ' ';
        }
        out << '\n';
    }
    return out.good();
}

static void configure_multicore_env(int requested_cores)
{
    if (requested_cores <= 0) return;

    const std::string cores_str = std::to_string(requested_cores);
    setenv("ACC_NUM_CORES", cores_str.c_str(), 1);
    setenv("OMP_NUM_THREADS", cores_str.c_str(), 1);

    setenv("OMP_PROC_BIND", "true", 0);
    setenv("OMP_PLACES", "cores", 0);

    setenv("OMP_DYNAMIC", "false", 0);
}


int main(int argc, char* argv[])
{
    int N = SIZE;
    double eps = EPS;
    int max_iter = MAX_ITER;
    std::string output_path = "out.txt";
    int cores = 0;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("size", po::value<int>(&N), "Grid size N (NxN)")
        ("eps", po::value<double>(&eps), "Tolerance")
        ("iters", po::value<int>(&max_iter), "Maximum iterations")
        ("cores", po::value<int>(&cores)->default_value(0), "CPU threads for -acc=multicore (0=do not touch env)")
        ("output", po::value<std::string>(&output_path)->default_value(output_path), "Output file for resulting matrix (text)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (std::exception& e) {
        std::cerr << "Error parsing command line: " << e.what() << std::endl;
        return 1;
    }

    if (N <= 9) {
        std::cerr << "Grid size must be greater than 9." << std::endl;
        return 1;
    }
    if (eps <= 0) {
        std::cerr << "Tolerance must be positive." << std::endl;
        return 1;
    }
    if (max_iter <= 0) {
        std::cerr << "Maximum iterations must be positive." << std::endl;
        return 1;
    }
    if (vm.count("help")) {
        std::cout << desc << std::endl;
    return 0;
}

    #if defined(ACC_MULTICORE_DEFAULTS)
    configure_multicore_env(cores);
    #endif
    double max_error = 0;
    
    int ny = N;
    int nx = N;
    int left_top = 10;
    int left_bottom = 20;
    int right_top = 20;
    int right_bottom = 30;
    double* __restrict__ local_grid = new double[ny * nx]();
    double* __restrict__ local_newgrid = new double[ny * nx]();

    double interpolation_value_top = (double)(right_top - left_top) / (double)(nx-1);
    double interpolation_value_botton = (right_bottom - left_bottom) / (double)(nx-1);
    for (int j = 0; j < nx - 1; j++) {
        int ind = IND(ny - 1, j);
       
        local_newgrid[j] = local_grid[j] = left_top + j * interpolation_value_top;
        local_newgrid[ind] = local_grid[ind] = left_bottom + j * interpolation_value_botton;
    }
    // Initialize bottom border
    // for (int j = 0; j < nx - 1; j++) {
    //     int ind = IND(ny - 1, j);
    //     double interpolation_value = (right_bottom - left_bottom + 1) / (double)ny;
    //     local_newgrid[ind] = local_grid[ind] = left_bottom + j * interpolation_value;
    // }

    double interpolation_value_l = (left_bottom - left_top) / (double)(ny-1);
    double interpolation_value_r = (right_bottom - right_top) / (double)(ny-1);
    for (int j = 0; j < ny - 1; j++) {
        int ind = IND(j, 0);
        int ind2 = IND(j, nx - 1);

        local_newgrid[ind] = local_grid[ind] = left_top + j * interpolation_value_l;
        local_newgrid[ind2] = local_grid[ind2] = right_top + j * interpolation_value_r;
    }
    local_grid[ny * nx - 1] = 30;
    local_newgrid[ny * nx - 1] = 30;
    // std::cout<<interpolation_value_top<<" "<<interpolation_value_botton<<std::endl;

    int iter = 0;
    auto start = std::chrono::steady_clock::now();

    #pragma acc data copy(local_grid[0:ny * nx]) create(local_newgrid[0:ny * nx])
    {
        for (;;) {
            double maxdiff = 0.0;
            iter++;
            if (iter > max_iter) break;

            #pragma acc parallel loop collapse(2) gang vector present(local_grid, local_newgrid) reduction(max:maxdiff)
            for (int i = 1; i < ny - 1; i++) {
                for (int j = 1; j < nx - 1; j++) {
                    int ind = i * nx + j;
                    local_newgrid[ind] = (local_grid[ind - nx] + local_grid[ind + nx] +
                                          local_grid[ind - 1] + local_grid[ind + 1]) * 0.25;
                    double diff = local_grid[ind] - local_newgrid[ind];
                    if (diff < 0) diff = -diff;
                    if (diff > maxdiff) maxdiff = diff;
                }
            }

            #pragma acc parallel loop collapse(2) gang vector present(local_grid, local_newgrid)
            for (int i = 1; i < ny - 1; i++) {
                for (int j = 1; j < nx - 1; j++) {
                    local_grid[i * nx + j] = local_newgrid[i * nx + j];
                }
            }

            max_error = maxdiff;
            if (maxdiff < eps) break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout<<"error: "<<max_error<<std::endl;
    std::chrono::duration<double> elapsed = end - start;
    std::cout<<"time: "<<elapsed.count()<<"\niterations: "<<iter<<std::endl;

    if (N == 10 || N == 13) {
        std::cout<<"\nFinal grid: (" <<N<<"x"<<N<<"):\n";
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++)
            {
                std::cout<<local_grid[i * nx + j]<<' ';
            }
            std::cout<<std::endl;
        }
    }

    if (!write_matrix_text(output_path, local_grid, N)) {
        std::cerr << "Failed to write matrix to: " << output_path << std::endl;
        delete[] local_grid;
        delete[] local_newgrid;
        return 2;
    }
    std::cout << "matrix_file: " << output_path << std::endl;
    
    delete[] local_grid;
    delete[] local_newgrid;
    return 0;
    }
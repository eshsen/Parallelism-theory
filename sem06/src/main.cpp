#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <memory>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

// Инициализация граничных условий методом линейной интерполяции
// Углы: top-left=10, top-right=20, bottom-right=30, bottom-left=20
void init_grid(double* grid, int N) {
    double top_left     = 10.0;
    double top_right    = 20.0;
    double bottom_right = 30.0;
    double bottom_left  = 20.0;

    // Заполнить внутреннюю область нулями
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            grid[i * N + j] = 0.0;
        }
    }

    // Верхняя граница (i=0): интерполяция top_left -> top_right
    for (int j = 0; j < N; j++) {
        grid[0 * N + j] = top_left + (top_right - top_left) * j / (N - 1);
    }

    // Нижняя граница (i=N-1): интерполяция bottom_left -> bottom_right
    for (int j = 0; j < N; j++) {
        grid[(N - 1) * N + j] = bottom_left + (bottom_right - bottom_left) * j / (N - 1);
    }

    // Левая граница (j=0): интерполяция top_left -> bottom_left
    for (int i = 0; i < N; i++) {
        grid[i * N + 0] = top_left + (bottom_left - top_left) * i / (N - 1);
    }

    // Правая граница (j=N-1): интерполяция top_right -> bottom_right
    for (int i = 0; i < N; i++) {
        grid[i * N + (N - 1)] = top_right + (bottom_right - top_right) * i / (N - 1);
    }
}

// Сохранение матрицы в файл
void save_matrix(const double* grid, int N, const std::string& filename) {
    std::ofstream f(filename);
    f << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            f << grid[i * N + j];
            if (j < N - 1) f << " ";
        }
        f << "\n";
    }
    f.close();
    std::cout << "Matrix saved to " << filename << std::endl;
}

// Вывод матрицы в терминал (для малых размеров)
void print_matrix(const double* grid, int N) {
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(9) << grid[i * N + j];
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    int    N          = 256;
    double accuracy   = 1e-6;
    int    max_iter   = 100;
    bool   print_mat  = false;
    bool   save_mat   = false;
    std::string out_file = "result.txt";

    po::options_description desc("Heat equation solver options");
    desc.add_options()
        ("help,h",                                          "Show help")
        ("size,N",    po::value<int>(&N)->default_value(256),
                      "Grid size N (NxN)")
        ("accuracy,a",po::value<double>(&accuracy)->default_value(1e-6),
                      "Target accuracy (epsilon)")
        ("iter,i",    po::value<int>(&max_iter)->default_value(100),
                      "Max iterations")
        ("print,p",   po::bool_switch(&print_mat),
                      "Print matrix to terminal (recommended N<=13)")
        ("save,s",    po::bool_switch(&save_mat),
                      "Save result matrix to file")
        ("output,o",  po::value<std::string>(&out_file)->default_value("result.txt"),
                      "Output file name");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    std::cout << "Grid size: " << N << "x" << N << "\n";
    std::cout << "Target accuracy: " << accuracy << "\n";
    std::cout << "Max iterations: " << max_iter << "\n";

    // Выделение памяти
    size_t sz = (size_t)N * N;
    std::unique_ptr<double[]> A(new double[sz]);
    std::unique_ptr<double[]> B(new double[sz]);

    double* a = A.get();
    double* b = B.get();

    init_grid(a, N);
    init_grid(b, N);

    double error = 0.0;
    int    iter  = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Основной итерационный цикл метода Якоби (пятиточечный шаблон)
    #pragma acc data copy(a[0:sz]) copyin(b[0:sz])
    {
        while (error > accuracy || iter == 0) {
            error = 0.0;

            #pragma acc parallel loop collapse(2) reduction(max:error) present(a[0:sz], b[0:sz])
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    b[i * N + j] = 0.25 * (a[(i - 1) * N + j]
                                         + a[(i + 1) * N + j]
                                         + a[i * N + (j - 1)]
                                         + a[i * N + (j + 1)]);
                    double diff = std::fabs(b[i * N + j] - a[i * N + j]);
                    error = (diff > error) ? diff : error;
                }
            }

            // Копирование b -> a (только внутренних точек)
            #pragma acc parallel loop collapse(2) present(a[0:sz], b[0:sz])
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    a[i * N + j] = b[i * N + j];
                }
            }

            iter++;
            if (iter >= max_iter) break;
        }
    } // конец acc data - данные копируются обратно в host

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Iterations: " << iter << "\n";
    std::cout << "Error:      " << std::scientific << error << "\n";
    std::cout << "Time:       " << std::fixed << std::setprecision(4)
              << elapsed << " s\n";

    if (print_mat) {
        std::cout << "\nResult matrix (" << N << "x" << N << "):\n";
        print_matrix(a, N);
    }

    if (save_mat) {
        save_matrix(a, N, out_file);
    }

    return 0;
}

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename T>
T func_sin(T x) {
    return static_cast<T>(std::sin(x));
}

template <typename T>
T func_sqrt(T x) {
    return static_cast<T>(std::sqrt(x));
}

template <typename T>
T func_pow(T x, T y) {
    return static_cast<T>(std::pow(x, y));
}

template <typename Tuple>
auto tuple_to_vector(const Tuple& t) {
    return std::apply([](auto&&... args) {
        return std::vector<double>{static_cast<double>(args)...};
    }, t);
}

template <typename T>
class Server {
private:
    using TaskWrapper = std::function<T()>;

    std::atomic<bool> stop_flag{false};
    std::atomic<std::size_t> next_task_id{0};

    std::queue<std::pair<std::size_t, TaskWrapper>> tasks;
    std::mutex tasks_mutex;
    std::condition_variable tasks_cv;

    std::unordered_map<std::size_t, T> results;
    std::mutex results_mutex;
    std::condition_variable results_cv;

    std::thread worker;

    void worker_loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(tasks_mutex);
            tasks_cv.wait(lock, [this] {
                return stop_flag.load() || !tasks.empty();
            });

            if (stop_flag.load() && tasks.empty()) {
                break;
            }

            auto [task_id, task] = std::move(tasks.front());
            tasks.pop();
            lock.unlock();

            T value = task();

            {
                std::lock_guard<std::mutex> res_lock(results_mutex);
                results[task_id] = value;
            }
            results_cv.notify_all();
        }
    }

public:
    void start() {
        if (worker.joinable()) {
            return;
        }
        stop_flag = false;
        worker = std::thread(&Server::worker_loop, this);
        std::cout << "[Server] started\n";
    }

    void stop() {
        stop_flag = true;
        tasks_cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        std::cout << "[Server] stopped\n";
    }

    template <typename Func, typename... Args>
    std::size_t add_task(Func&& func, Args&&... args) {
        auto args_tuple = std::make_tuple(std::forward<Args>(args)...);
        TaskWrapper wrapped = [func, args_tuple]() -> T {
            return std::apply(func, args_tuple);
        };

        const std::size_t id = ++next_task_id;
        {
            std::lock_guard<std::mutex> lock(tasks_mutex);
            tasks.emplace(id, std::move(wrapped));
        }
        tasks_cv.notify_one();
        return id;
    }

    T request_result(std::size_t id) {
        std::unique_lock<std::mutex> lock(results_mutex);
        results_cv.wait(lock, [this, id] {
            return stop_flag.load() || results.find(id) != results.end();
        });
        return results.at(id);
    }
};

double random_double(double left, double right) {
    thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(left, right);
    return dist(gen);
}

template <typename T, typename Func, typename Generator>
void client_worker(Server<T>& server,
                   int client_id,
                   int func_id,
                   Func func,
                   Generator generator,
                   const std::string& filename,
                   int n_tasks) {
    std::vector<std::size_t> ids;
    std::vector<std::vector<double>> args_store;
    ids.reserve(n_tasks);
    args_store.reserve(n_tasks);

    for (int i = 0; i < n_tasks; ++i) {
        auto args_tuple = generator();
        args_store.push_back(tuple_to_vector(args_tuple));

        std::size_t id = std::apply([&](auto&&... unpacked) {
            return server.add_task(func, unpacked...);
        }, args_tuple);

        ids.push_back(id);
    }

    std::ofstream out(filename);
    for (int i = 0; i < n_tasks; ++i) {
        T result = server.request_result(ids[i]);
        out << ids[i] << ' ' << func_id << ' ' << args_store[i].size();
        for (double arg : args_store[i]) {
            out << ' ' << arg;
        }
        out << ' ' << result << '\n';
    }

    std::cout << "[Client " << client_id << "] finished -> " << filename << "\n";
}

int main() {
    constexpr int N = 9999;
    std::filesystem::create_directories("results");

    Server<double> server;
    server.start();

    auto sin_gen = []() {
        return std::make_tuple(random_double(0.0, 2.0 * M_PI));
    };

    auto sqrt_gen = []() {
        return std::make_tuple(random_double(0.0, 10000.0));
    };

    auto pow_gen = []() {
        return std::make_tuple(random_double(1.0, 10.0), random_double(1.0, 5.0));
    };

    std::thread client1(client_worker<double, decltype(func_sin<double>), decltype(sin_gen)>,
                        std::ref(server), 1, 1, func_sin<double>, sin_gen,
                        "results/client1_sin.txt", N);

    std::thread client2(client_worker<double, decltype(func_sqrt<double>), decltype(sqrt_gen)>,
                        std::ref(server), 2, 2, func_sqrt<double>, sqrt_gen,
                        "results/client2_sqrt.txt", N);

    std::thread client3(client_worker<double, decltype(func_pow<double>), decltype(pow_gen)>,
                        std::ref(server), 3, 3, func_pow<double>, pow_gen,
                        "results/client3_pow.txt", N);

    client1.join();
    client2.join();
    client3.join();

    server.stop();
    return 0;
}
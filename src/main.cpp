#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <random>
#include <iomanip> // for std::setprecision
#include <sys/stat.h> // for mkdir
#include "core/dataset.h"
#include "algorithms/pruning_algorithm.h"
#include "algorithms/kmeans_triangle_pruning.h"
#include "core/distance.h"

// 辅助函数：如果数据集不存在，则生成一个随机数据集
// 现在它会创建一个目录并将 nodes.txt 放入其中
void generate_random_data(const std::string& dir_path, int num_points, int dimensions) {
    struct stat info{};
    // 检查目录是否存在，不存在则创建
    if (stat(dir_path.c_str(), &info) != 0) {
        std::cout << "Generating random dataset directory: " << dir_path << std::endl;
        // 在Linux/macOS上使用 mkdir。在Windows上可能需要 _mkdir。
        // C++17 <filesystem> 是跨平台的最佳选择。
        #if defined(_WIN32)
            _mkdir(dir_path.c_str());
        #else
            mkdir(dir_path.c_str(), 0755);
        #endif
    }

    std::string nodes_filepath = dir_path + "/nodes.txt";
    std::ifstream f(nodes_filepath);
    if (f.good()) {
        return; // nodes.txt 文件已存在
    }

    std::cout << "Generating random nodes.txt in " << dir_path << std::endl;
    std::ofstream out(nodes_filepath);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < dimensions; ++j) {
            out << dis(gen) << (j == dimensions - 1 ? "" : " ");
        }
        out << "\n";
    }
}

int main() {
    // --- 实验参数 ---
    const std::string DATA_ROOT = "../data/";
    const std::string DATASET_NAME = "PubMed"; // <--- 在这里指定你要测试的数据集名称

    // 如果SinaNet不存在，则会生成并使用下面的随机数据集
    const std::string RANDOM_DATASET_NAME = "Random_10k_128d";

    constexpr int K_MEANS_K = 2000;
    constexpr int K_MEANS_ITERATIONS = 10;
    constexpr double QUERY_RADIUS = 0.5 ;
    constexpr int NUM_QUERIES = 100000;

    // --- 准备数据 ---
    Dataset dataset;
    std::string dataset_dir = DATA_ROOT + DATASET_NAME;

    // 尝试加载指定的数据集
    if (!dataset.load_from_directory(dataset_dir)) {
        constexpr int DIMENSIONS = 128;
        constexpr int NUM_POINTS = 10000;
        std::cout << "Could not load dataset '" << DATASET_NAME
                  << "'. Generating and using a random dataset for demonstration." << std::endl;

        dataset_dir = DATA_ROOT + RANDOM_DATASET_NAME;
        generate_random_data(dataset_dir, NUM_POINTS, DIMENSIONS);

        if (!dataset.load_from_directory(dataset_dir)) {
            std::cerr << "Fatal: Failed to load even the generated dataset. Exiting." << std::endl;
            return 1;
        }
    }

    // --- 初始化算法 ---
    std::unique_ptr<PruningAlgorithm> algorithm =
        std::make_unique<KMeansTrianglePruning>(K_MEANS_K, K_MEANS_ITERATIONS);

    // --- 1. 构建索引阶段 ---
    auto start_build = std::chrono::high_resolution_clock::now();
    algorithm->build(dataset);
    auto end_build = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> build_time = end_build - start_build;

    std::cout << "\n--- Build Phase ---" << std::endl;
    std::cout << "Build time: " << build_time.count() << " ms" << std::endl;

    // --- 2. 查询阶段 ---
    std::cout << "\n--- Query Phase ---" << std::endl;
    std::cout << "Running " << NUM_QUERIES << " queries with radius r = " << QUERY_RADIUS << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, dataset.size() - 1);

    algorithm->reset_stats();
    auto start_query = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_QUERIES; ++i) {
        int p_idx = distrib(gen);
        int q_idx = distrib(gen);
        if (p_idx == q_idx) continue;
        algorithm->query_distance_exceeds(p_idx, q_idx, QUERY_RADIUS);
    }
    auto end_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> query_time = end_query - start_query;

    // --- 3. 结果统计 ---
    long long full_calcs = algorithm->get_full_calculations_count();
    long long total_valid_queries = NUM_QUERIES; // 简化处理，可精确计算
    long long pruned_calcs = total_valid_queries - full_calcs;
    double pruning_rate = static_cast<double>(pruned_calcs) / total_valid_queries * 100.0;

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Total query time: " << query_time.count() << " ms" << std::endl;
    std::cout << "Average query time: " << query_time.count() / total_valid_queries << " ms" << std::endl;
    std::cout << "Total queries: " << total_valid_queries << std::endl;
    std::cout << "Full distance calculations: " << full_calcs << std::endl;
    std::cout << "Pruned queries: " << pruned_calcs << std::endl;
    std::cout << "Pruning Rate: " << std::fixed << std::setprecision(2) << pruning_rate << "%" << std::endl;

    return 0;
}
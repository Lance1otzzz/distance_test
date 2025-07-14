#include "multi_pivot_triangle_pruning.h"
#include "../core/distance.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath> // for std::abs

MultiPivotTrianglePruning::MultiPivotTrianglePruning(int k, int max_iterations)
    : k_(k), max_iterations_(max_iterations) {}

void MultiPivotTrianglePruning::build(const Dataset& dataset) {
    dataset_ = &dataset;
    std::cout << "Building index with Multi-Pivot Triangle Pruning (k=" << k_ << ")..." << std::endl;
    
    // 1. 运行 K-means 找到 k 个 pivots
    run_kmeans();

    // 2. 预计算每个点到所有 k 个 pivots 的距离
    std::cout << "Pre-calculating point-to-all-pivots distances..." << std::endl;
    size_t num_points = dataset_->size();
    precomputed_dists_.assign(num_points, std::vector<double>(k_));

    for (size_t i = 0; i < num_points; ++i) {
        for (int j = 0; j < k_; ++j) {
            precomputed_dists_[i][j] = euclidean_distance(dataset_->get_point(i), pivots_[j]);
        }
    }
    std::cout << "Build finished." << std::endl;
}

bool MultiPivotTrianglePruning::query_distance_exceeds(int p_idx, int q_idx, double r) {
    // --- A-La-Carte 三角不等式剪枝 ---
    // 核心思想：遍历所有k个pivot，找到最紧的下界
    
    double max_lower_bound = 0.0;

    // 查找p和q到所有pivots的预计算距离
    const auto& p_dists_to_pivots = precomputed_dists_[p_idx];
    const auto& q_dists_to_pivots = precomputed_dists_[q_idx];

    for (int i = 0; i < k_; ++i) {
        // 对于每个 pivot_i, 我们有 d(p,q) >= |d(p, pivot_i) - d(q, pivot_i)|
        double dist_p_to_pivot_i = p_dists_to_pivots[i];
        double dist_q_to_pivot_i = q_dists_to_pivots[i];
        
        double current_lower_bound = std::abs(dist_p_to_pivot_i - dist_q_to_pivot_i);
        
        if (current_lower_bound > max_lower_bound) {
            max_lower_bound = current_lower_bound;
        }
    }

    // 使用找到的最紧下界进行剪枝
    if (max_lower_bound > r) {
        // 剪枝成功，我们确定 d(p,q) > r
        return true;
    }

    // 如果最紧的下界都无法剪枝，我们还可以尝试使用上界剪枝。
    // 这里需要找到一个能提供最紧上界的pivot。
    // d(p,q) <= d(p, pivot_i) + d(q, pivot_i)
    // 我们要找使 d(p, pivot_i) + d(q, pivot_i) 最小的 pivot_i
    double min_upper_bound = std::numeric_limits<double>::max();
    for (int i = 0; i < k_; ++i) {
        double current_upper_bound = p_dists_to_pivots[i] + q_dists_to_pivots[i];
        if (current_upper_bound < min_upper_bound) {
            min_upper_bound = current_upper_bound;
        }
    }

    if (min_upper_bound <= r) {
        // 剪枝成功，我们确定 d(p,q) <= r
        return false;
    }

    // 剪枝失败，必须进行完整计算
    full_calculations_count_++;
    const auto& p = dataset_->get_point(p_idx);
    const auto& q = dataset_->get_point(q_idx);
    return euclidean_distance(p, q) > r;
}

// K-means 实现 (这部分代码与之前的 kmeans_triangle_pruning.cpp 完全相同)
void MultiPivotTrianglePruning::run_kmeans() {
    const auto& points = dataset_->get_all_points();
    size_t num_points = points.size();
    size_t dimensions = dataset_->dimensions();

    pivots_.clear();
    std::vector<int> initial_indices(num_points);
    std::iota(initial_indices.begin(), initial_indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(initial_indices.begin(), initial_indices.end(), g);
    for (int i = 0; i < k_; ++i) {
        pivots_.push_back(points[initial_indices[i]]);
    }

    std::vector<int> point_to_pivot_map(num_points, 0);

    for (int iter = 0; iter < max_iterations_; ++iter) {
        for (size_t i = 0; i < num_points; ++i) {
            double min_dist_sq = std::numeric_limits<double>::max();
            int best_pivot_idx = 0;
            for (int j = 0; j < k_; ++j) {
                double dist_sq = euclidean_distance_sq(points[i], pivots_[j]);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_pivot_idx = j;
                }
            }
            point_to_pivot_map[i] = best_pivot_idx;
        }

        std::vector<Point> new_pivots(k_, Point(dimensions, 0.0));
        std::vector<int> cluster_counts(k_, 0);
        for (size_t i = 0; i < num_points; ++i) {
            int pivot_idx = point_to_pivot_map[i];
            for (size_t d = 0; d < dimensions; ++d) {
                new_pivots[pivot_idx][d] += points[i][d];
            }
            cluster_counts[pivot_idx]++;
        }

        for (int j = 0; j < k_; ++j) {
            if (cluster_counts[j] > 0) {
                for (size_t d = 0; d < dimensions; ++d) {
                    new_pivots[j][d] /= cluster_counts[j];
                }
                pivots_[j] = new_pivots[j];
            }
        }
    }
}
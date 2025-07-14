#include "kmeans_triangle_pruning.h"
#include "../core/distance.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <limits>

KMeansTrianglePruning::KMeansTrianglePruning(int k, int max_iterations)
    : k_(k), max_iterations_(max_iterations) {}

void KMeansTrianglePruning::build(const Dataset& dataset) {
    dataset_ = &dataset;
    std::cout << "Building index with K-means (k=" << k_ << ")..." << std::endl;
    
    // 运行 K-means 找到 pivots
    run_kmeans();

    // 预计算每个点到其中心点的距离
    std::cout << "Pre-calculating point-to-pivot distances..." << std::endl;
    point_to_pivot_dist_.resize(dataset_->size());
    for (size_t i = 0; i < dataset_->size(); ++i) {
        int pivot_idx = point_to_pivot_map_[i];
        point_to_pivot_dist_[i] = euclidean_distance(dataset_->get_point(i), pivots_[pivot_idx]);
    }
    std::cout << "Build finished." << std::endl;
}

bool KMeansTrianglePruning::query_distance_exceeds(int p_idx, int q_idx, double r) {
    // 获取点p, q的信息
    int pivot_p_idx = point_to_pivot_map_[p_idx];
    int pivot_q_idx = point_to_pivot_map_[q_idx];
    double dist_p_to_pivot = point_to_pivot_dist_[p_idx];
    double dist_q_to_pivot = point_to_pivot_dist_[q_idx];

    // 三角不等式剪枝
    // 设 d(x,y) 为距离
    // 1. 下界: d(p,q) >= |d(p, pivot_p) - d(q, pivot_p)|
    //    更有用的下界: d(p,q) >= d(pivot_p, pivot_q) - d(p, pivot_p) - d(q, pivot_q)
    // 2. 上界: d(p,q) <= d(p, pivot_p) + d(pivot_p, pivot_q) + d(q, pivot_q)
    
    double dist_pivots = euclidean_distance(pivots_[pivot_p_idx], pivots_[pivot_q_idx]);
    
    // 规则1 (基于下界): 如果 d(pivots) - d(p,pivot_p) - d(q,pivot_q) > r，那么 d(p,q) 必定 > r
    if (dist_pivots - dist_p_to_pivot - dist_q_to_pivot > r) {
        return true; // 剪枝成功，返回 true
    }

    // 规则2 (基于上界): 如果 d(pivots) + d(p,pivot_p) + d(q,pivot_q) <= r, 那么 d(p,q) 必定 <= r
    if (dist_pivots + dist_p_to_pivot + dist_q_to_pivot <= r) {
        return false; // 剪枝成功，返回 false
    }

    // 剪枝失败，必须进行完整计算
    full_calculations_count_++;
    const auto& p = dataset_->get_point(p_idx);
    const auto& q = dataset_->get_point(q_idx);
    return euclidean_distance(p, q) > r;
}

// 内部 K-means 实现
void KMeansTrianglePruning::run_kmeans() {
    const auto& points = dataset_->get_all_points();
    size_t num_points = points.size();
    size_t dimensions = dataset_->dimensions();

    // 1. 初始化中心点 (随机选择k个点)
    pivots_.clear();
    std::vector<int> initial_indices(num_points);
    std::iota(initial_indices.begin(), initial_indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(initial_indices.begin(), initial_indices.end(), g);
    for (int i = 0; i < k_; ++i) {
        pivots_.push_back(points[initial_indices[i]]);
    }

    point_to_pivot_map_.assign(num_points, 0);

    for (int iter = 0; iter < max_iterations_; ++iter) {
        // 2. 分配步骤: 将每个点分配给最近的中心点
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
            point_to_pivot_map_[i] = best_pivot_idx;
        }

        // 3. 更新步骤: 重新计算每个簇的中心点
        std::vector<Point> new_pivots(k_, Point(dimensions, 0.0));
        std::vector<int> cluster_counts(k_, 0);
        for (size_t i = 0; i < num_points; ++i) {
            int pivot_idx = point_to_pivot_map_[i];
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
         // 简单的收敛判断可以加在这里，比如看中心点移动距离，但为简单起见，我们固定迭代次数
    }
}
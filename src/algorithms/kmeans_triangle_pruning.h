#pragma once
#include "pruning_algorithm.h"
#include <vector>

class KMeansTrianglePruning final : public PruningAlgorithm {
public:
    KMeansTrianglePruning(int k, int max_iterations);

    void build(const Dataset& dataset) override;
    bool query_distance_exceeds(int p_idx, int q_idx, double r) override;

private:
    // K-means 算法实现
    void run_kmeans();

    int k_;
    int max_iterations_;
    const Dataset* dataset_ = nullptr; // 指向原始数据集

    std::vector<Point> pivots_; // k个中心点 (pivots/centroids)
    std::vector<int> point_to_pivot_map_; // 每个点属于哪个中心
    std::vector<double> point_to_pivot_dist_; // 每个点到其所属中心的距离
};
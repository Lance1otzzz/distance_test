#pragma once
#include "pruning_algorithm.h"
#include <vector>

class MultiPivotTrianglePruning : public PruningAlgorithm {
public:
    MultiPivotTrianglePruning(int k, int max_iterations);

    void build(const Dataset& dataset) override;
    bool query_distance_exceeds(int p_idx, int q_idx, double r) override;

private:
    // K-means 算法实现 (与之前相同)
    void run_kmeans();

    int k_;
    int max_iterations_;
    const Dataset* dataset_ = nullptr;

    std::vector<Point> pivots_; // k个中心点 (pivots/centroids)
    
    // 关键改动：存储每个点到所有k个pivot的距离
    // precomputed_dists_[i][j] = distance(point_i, pivot_j)
    std::vector<std::vector<double>> precomputed_dists_; 
};
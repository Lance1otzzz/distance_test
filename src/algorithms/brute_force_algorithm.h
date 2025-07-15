#pragma once
#include "pruning_algorithm.h"

class BruteForceAlgorithm : public PruningAlgorithm {
public:
    BruteForceAlgorithm() = default;

    // build 方法是空的，因为暴力搜索不需要预处理
    void build(const Dataset& dataset) override;

    // query 方法总是执行完整的距离计算
    bool query_distance_exceeds(int p_idx, int q_idx, double r) override;

private:
    const Dataset* dataset_ = nullptr;
};
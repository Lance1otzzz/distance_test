#include "brute_force_algorithm.h"
#include "../core/distance.h"

void BruteForceAlgorithm::build(const Dataset& dataset) {
    // 无需构建任何东西，但需要保存数据集指针
    dataset_ = &dataset;
}

bool BruteForceAlgorithm::query_distance_exceeds(int p_idx, int q_idx, double r) {
    // 暴力算法总是进行“完整”计算
    full_calculations_count_++;
    
    const auto& p = dataset_->get_point(p_idx);
    const auto& q = dataset_->get_point(q_idx);

    // 使用我们最高效的“完整”计算方法
    return is_distance_exceeding_early_exit(p, q, r);
}
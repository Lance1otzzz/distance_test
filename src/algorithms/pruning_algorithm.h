#pragma once
#include "../core/dataset.h"

class PruningAlgorithm {
public:
    virtual ~PruningAlgorithm() = default;

    // 预处理/构建索引的方法
    virtual void build(const Dataset& dataset) = 0;

    // 查询接口：判断 p_idx 和 q_idx 两点的距离是否超过 r
    // 返回 true 表示 dist > r, false 表示 dist <= r
    virtual bool query_distance_exceeds(int p_idx, int q_idx, double r) = 0;

    // 获取统计信息：完整计算的次数
    [[nodiscard]] long long get_full_calculations_count() const { return full_calculations_count_; }
    void reset_stats() { full_calculations_count_ = 0; }

protected:
    long long full_calculations_count_ = 0; // 用于统计剪枝失败、必须进行完整计算的次数
};
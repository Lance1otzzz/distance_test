#pragma once
#include "point.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

// 计算两点之间欧氏距离的平方（避免开方，速度更快）
inline double euclidean_distance_sq(const Point& p1, const Point& p2) {
    if (p1.size() != p2.size()) {
        throw std::invalid_argument("Points must have the same dimension.");
    }
    double sum_sq = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum_sq += diff * diff;
    }
    return sum_sq;
}

// 计算两点之间的欧氏距离
inline double euclidean_distance(const Point& p1, const Point& p2) {
    return std::sqrt(euclidean_distance_sq(p1, p2));
}
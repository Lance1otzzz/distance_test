#pragma once
#include "point.h"
#include <string>
#include <vector>

class Dataset {
public:
    // 从指定的数据集目录加载数据 (会寻找目录下的 nodes.txt)
    bool load_from_directory(const std::string& dir_path);

    [[nodiscard]] const Point& get_point(int index) const { return points_[index]; }
    [[nodiscard]] size_t size() const { return points_.size(); }
    [[nodiscard]] size_t dimensions() const { return dimensions_; }

    [[nodiscard]] const std::vector<Point>& get_all_points() const { return points_; }

private:
    std::vector<Point> points_;
    size_t dimensions_ = 0;
};
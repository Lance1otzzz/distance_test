#include "dataset.h"
#include <fstream>
#include <iostream>
#include <sstream>

bool Dataset::load_from_directory(const std::string& dir_path) {
    // 拼接出 nodes.txt 的完整路径
    // 注意：在 Windows 上，路径分隔符可能是 '\'。为简单起见，这里使用 '/'，
    // 在现代操作系统和C++库中通常可以通用。
    // 使用 C++17 的 <filesystem> 是更健壮的做法。
    std::string nodes_filepath = dir_path + "/nodes.txt";

    std::ifstream file(nodes_filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open nodes file: " << nodes_filepath << std::endl;
        return false;
    }

    // 清空旧数据，以便重用 Dataset 对象
    points_.clear();
    dimensions_ = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        Point p;
        double val;
        while (ss >> val) {
            p.push_back(val);
        }

        if (!p.empty()) {
            if (points_.empty()) {
                dimensions_ = p.size();
            } else {
                if (p.size() != dimensions_) {
                    std::cerr << "Error: Inconsistent dimension in dataset file." << std::endl;
                    points_.clear(); // 加载失败，清空
                    return false;
                }
            }
            points_.push_back(p);
        }
    }

    std::cout << "Dataset '" << dir_path << "' loaded: " << points_.size() << " points, "
              << dimensions_ << " dimensions." << std::endl;
    return true;
}
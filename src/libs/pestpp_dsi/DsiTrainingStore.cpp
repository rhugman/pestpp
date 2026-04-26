#include "DsiTrainingStore.h"

#include <stdexcept>
#include <unordered_map>

namespace pestpp_dsi {

DsiTrainingStore::DsiTrainingStore() = default;

void DsiTrainingStore::append(const Eigen::MatrixXd& X,
                              const std::vector<std::string>& obs_names) {
    if (X.cols() != static_cast<Eigen::Index>(obs_names.size())) {
        throw std::invalid_argument(
            "DsiTrainingStore::append: X.cols() != obs_names.size()");
    }
    if (obs_names_.empty()) {
        obs_names_ = obs_names;
        store_ = X;
        return;
    }
    if (obs_names != obs_names_) {
        throw std::invalid_argument(
            "DsiTrainingStore::append: obs_names differ from stored order");
    }
    const Eigen::Index old_rows = store_.rows();
    Eigen::MatrixXd combined(old_rows + X.rows(), store_.cols());
    combined.topRows(old_rows) = store_;
    combined.bottomRows(X.rows()) = X;
    store_ = std::move(combined);
}

Eigen::MatrixXd DsiTrainingStore::matrix_for(
    const std::vector<std::string>& obs_names) const {
    if (obs_names == obs_names_) return store_;
    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx.reserve(obs_names_.size());
    for (int i = 0; i < static_cast<int>(obs_names_.size()); ++i) {
        name_to_idx.emplace(obs_names_[i], i);
    }
    Eigen::MatrixXd out(store_.rows(), obs_names.size());
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(obs_names.size()); ++j) {
        auto it = name_to_idx.find(obs_names[j]);
        if (it == name_to_idx.end()) {
            throw std::invalid_argument(
                "DsiTrainingStore::matrix_for: unknown obs name '"
                + obs_names[j] + "'");
        }
        out.col(j) = store_.col(it->second);
    }
    return out;
}

void DsiTrainingStore::clear() {
    store_.resize(0, 0);
    obs_names_.clear();
}

}  // namespace pestpp_dsi

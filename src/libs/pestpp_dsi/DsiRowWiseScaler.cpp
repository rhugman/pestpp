#include "DsiRowWiseScaler.h"

#include <algorithm>
#include <stdexcept>

namespace pestpp_dsi {

DsiRowWiseScaler::DsiRowWiseScaler()
    : feature_range_({-1.0, 1.0}) {}

DsiRowWiseScaler::DsiRowWiseScaler(std::pair<double, double> feature_range,
                                   GroupCols groups,
                                   GroupCols fit_groups)
    : feature_range_(std::move(feature_range)),
      groups_(std::move(groups)),
      fit_groups_(std::move(fit_groups)) {
    if (fit_groups_.empty()) {
        // pyemu default: fit_groups defaults to groups (same columns).
        fit_groups_ = groups_;
    }
}

int DsiRowWiseScaler::n_fit_rows() const {
    if (!fitted_ || row_min_.empty()) return 0;
    return static_cast<int>(row_min_.begin()->second.size());
}

void DsiRowWiseScaler::fit(const Eigen::MatrixXd& X) {
    row_min_.clear();
    row_max_.clear();
    if (groups_.empty()) {
        throw std::invalid_argument(
            "DsiRowWiseScaler::fit: no groups configured");
    }
    for (const auto& kv : groups_) {
        const std::string& name = kv.first;
        // fit_cols: pyemu falls back to group cols when fit_groups
        // doesn't contain this group (transformers.py:118).
        std::vector<int> fit_cols;
        auto it = fit_groups_.find(name);
        if (it != fit_groups_.end()) {
            fit_cols = it->second;
        } else {
            fit_cols = kv.second;
        }
        if (fit_cols.empty()) continue;

        Eigen::VectorXd rmin = Eigen::VectorXd::Zero(X.rows());
        Eigen::VectorXd rmax = Eigen::VectorXd::Zero(X.rows());
        for (Eigen::Index r = 0; r < X.rows(); ++r) {
            double mn = std::numeric_limits<double>::infinity();
            double mx = -std::numeric_limits<double>::infinity();
            for (int c : fit_cols) {
                const double v = X(r, c);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            rmin(r) = mn;
            rmax(r) = mx;
        }
        row_min_[name] = std::move(rmin);
        row_max_[name] = std::move(rmax);
    }
    fitted_ = true;
}

void DsiRowWiseScaler::fill_row_params(const std::string& group_name,
                                       Eigen::VectorXd& row_min,
                                       Eigen::VectorXd& row_max,
                                       Eigen::VectorXd& row_range) const {
    row_min = row_min_.at(group_name);
    row_max = row_max_.at(group_name);
    row_range = row_max - row_min;
    // Constant-row protection (transformers.py:163).
    for (Eigen::Index i = 0; i < row_range.size(); ++i) {
        if (row_range(i) == 0.0) row_range(i) = 1.0;
    }
}

namespace {

// Broadcast helper: returns the value to use at output row r given a
// fit-time vector v whose length is either 1 (truth-scaler broadcast)
// or n_rows (training-scaler elementwise).
inline double broadcast_at(const Eigen::VectorXd& v, Eigen::Index r,
                           Eigen::Index n_rows) {
    if (v.size() == 1) return v(0);
    if (v.size() != n_rows) {
        throw std::invalid_argument(
            "DsiRowWiseScaler: row-param size mismatch with input");
    }
    return v(r);
}

}  // namespace

void DsiRowWiseScaler::apply(Eigen::MatrixXd& X) const {
    if (!fitted_) {
        throw std::logic_error("DsiRowWiseScaler::apply called before fit");
    }
    const double f_min = feature_range_.first;
    const double f_max = feature_range_.second;
    const double f_span = f_max - f_min;
    for (const auto& kv : groups_) {
        const std::string& name = kv.first;
        const std::vector<int>& cols = kv.second;
        if (cols.empty()) continue;
        Eigen::VectorXd rmin, rmax, rrange;
        fill_row_params(name, rmin, rmax, rrange);
        for (Eigen::Index r = 0; r < X.rows(); ++r) {
            const double mn = broadcast_at(rmin, r, X.rows());
            const double rr = broadcast_at(rrange, r, X.rows());
            for (int c : cols) {
                const double std_val = (X(r, c) - mn) / rr;   // [0, 1]
                X(r, c) = std_val * f_span + f_min;            // -> feature
            }
        }
    }
}

void DsiRowWiseScaler::inverse(Eigen::MatrixXd& X) const {
    if (!fitted_) {
        throw std::logic_error("DsiRowWiseScaler::inverse called before fit");
    }
    const double f_min = feature_range_.first;
    const double f_max = feature_range_.second;
    const double f_span = f_max - f_min;
    for (const auto& kv : groups_) {
        const std::string& name = kv.first;
        const std::vector<int>& cols = kv.second;
        if (cols.empty()) continue;
        Eigen::VectorXd rmin, rmax, rrange;
        fill_row_params(name, rmin, rmax, rrange);
        for (Eigen::Index r = 0; r < X.rows(); ++r) {
            const double mn = broadcast_at(rmin, r, X.rows());
            const double rr = broadcast_at(rrange, r, X.rows());
            for (int c : cols) {
                const double std_val = (X(r, c) - f_min) / f_span;
                X(r, c) = std_val * rr + mn;
            }
        }
    }
}

}  // namespace pestpp_dsi

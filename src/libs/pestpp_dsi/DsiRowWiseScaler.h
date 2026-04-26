// DsiRowWiseScaler.h
//
// Per-row min-max scaler grouped by observation subsets. Faithful
// port of pyemu.emulators.transformers.RowWiseMinMaxScaler
// (transformers.py:72-215).
//
// Two flavours:
//   - "training" scaler: fit on (n_real x n_obs); row_min / row_max
//     are vectors of length n_real per group; apply scales each row
//     using its own min/max.
//   - "truth" scaler: fit on a single-row truth vector (1 x n_obs);
//     row_min / row_max are scalars per group; apply broadcasts to
//     all rows of the input. This is the "pattern-DSI" inverse used
//     during predict() to map SVD output (in feature_range space)
//     back to original units (plan §4.3 step 2).

#ifndef PESTPP_DSI_ROWWISE_SCALER_H
#define PESTPP_DSI_ROWWISE_SCALER_H

#include <Eigen/Dense>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace pestpp_dsi {

class DsiRowWiseScaler {
public:
    using GroupCols = std::map<std::string, std::vector<int>>;

    DsiRowWiseScaler();
    DsiRowWiseScaler(std::pair<double, double> feature_range,
                     GroupCols groups,
                     GroupCols fit_groups);

    void fit(const Eigen::MatrixXd& X);
    void apply(Eigen::MatrixXd& X) const;
    void inverse(Eigen::MatrixXd& X) const;

    bool is_fitted() const { return fitted_; }
    int n_groups() const { return static_cast<int>(groups_.size()); }

    // Number of rows of fit data (1 for the truth-scaler case,
    // n_real for the training case). Apply broadcasts when this is 1.
    int n_fit_rows() const;

    const GroupCols& groups() const { return groups_; }
    const std::pair<double, double>& feature_range() const {
        return feature_range_;
    }

private:
    void fill_row_params(const std::string& group_name,
                         Eigen::VectorXd& row_min,
                         Eigen::VectorXd& row_max,
                         Eigen::VectorXd& row_range) const;

    bool fitted_ = false;
    std::pair<double, double> feature_range_;
    GroupCols groups_;
    GroupCols fit_groups_;
    std::map<std::string, Eigen::VectorXd> row_min_;
    std::map<std::string, Eigen::VectorXd> row_max_;
};

}  // namespace pestpp_dsi

#endif  // PESTPP_DSI_ROWWISE_SCALER_H

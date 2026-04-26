// DsiTrainingStore.h
//
// Append-only accumulator for FOM-evaluated observation ensembles
// across IES iterations. Lambda-loop integration (Phase 3) appends
// the post-FOM `oe` rows after the remainder sweep at
// EnsembleMethodUtils.cpp:7588 so the surrogate refits on a growing
// posterior-aware sample.

#ifndef PESTPP_DSI_TRAINING_STORE_H
#define PESTPP_DSI_TRAINING_STORE_H

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace pestpp_dsi {

class DsiTrainingStore {
public:
    DsiTrainingStore();

    // Append rows of training data. The first append seeds the
    // column order; subsequent appends must use the same obs_names
    // (same length, same order) — throws std::invalid_argument
    // otherwise. Lambda-loop callers always pass act_obs_names_nnz,
    // which is fixed for a run.
    //
    // X: (n_new_real x n_obs).
    void append(const Eigen::MatrixXd& X,
                const std::vector<std::string>& obs_names);

    // Snapshot of the accumulated matrix in stored column order.
    const Eigen::MatrixXd& matrix() const { return store_; }

    // Snapshot reordered to the requested obs_names. Throws if a
    // requested name was never appended.
    Eigen::MatrixXd matrix_for(
        const std::vector<std::string>& obs_names) const;

    int n_realisations() const { return static_cast<int>(store_.rows()); }
    int n_obs() const { return static_cast<int>(store_.cols()); }
    const std::vector<std::string>& obs_names() const { return obs_names_; }

    void clear();

private:
    Eigen::MatrixXd store_;          // (n_real_total x n_obs)
    std::vector<std::string> obs_names_;
};

}  // namespace pestpp_dsi

#endif  // PESTPP_DSI_TRAINING_STORE_H

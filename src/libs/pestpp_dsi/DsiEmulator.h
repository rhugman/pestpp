// DsiEmulator.h
//
// Native C++ Data Space Inversion (DSI) emulator. Faithful port of
// pyemu.emulators.DSI (dependencies/pyemu/pyemu/emulators/dsi.py).
//
// Lifecycle: construct with Config -> fit(training_obs) -> predict(latent)
// or project_oe(obs_ensemble). The fit step computes the projection
// matrix `pmat = V * diag(s)` from a compact SVD of the centered,
// (N-1)-scaled, transformed training data and retains components up
// to `energy_threshold`.
//
// Phase 1 ships Identity and Log10 transforms only. NormalScore,
// rowwise scaling, serialization (save/load) land in later phases.

#ifndef PESTPP_DSI_EMULATOR_H
#define PESTPP_DSI_EMULATOR_H

#include "DsiRowWiseScaler.h"
#include "DsiTransformPipeline.h"

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pestpp_dsi {

class DsiEmulator {
public:
    struct TransformSpec {
        enum class Kind {
            Identity,
            Log10,
            NormalScore,         // Phase 2
            NormalScoreQuad      // Phase 2
        };
        Kind kind = Kind::Identity;
        // Subset of obs_names this transform acts on. Empty == all.
        std::vector<std::string> columns;
    };

    struct Config {
        // Column order (and identity) of the training data matrix.
        std::vector<std::string> obs_names;

        // Optional metadata; obs_truth is required when rowwise_groups
        // is non-empty (the truth-row scaler is fit from it).
        std::vector<double> obs_weights;
        std::vector<double> obs_truth;

        // Forward-transform stages. Order matters: applied left to
        // right at fit/apply, reversed at inverse.
        std::vector<TransformSpec> transforms;

        // SVD energy retention threshold. >= 1.0 keeps all components.
        double energy_threshold = 0.99;

        // Pattern-DSI rowwise scaling. Maps group name to obs-name list.
        // When set:
        //   - fit() fits a training rowwise scaler on the post-transform
        //     matrix and applies it before the SVD (plan §4.2 step 2).
        //   - fit() also fits a truth rowwise scaler on cfg.obs_truth
        //     (post-transform); predict() uses its inverse to map SVD
        //     output back to original units (plan §4.2 step 10, §4.3
        //     step 2).
        std::map<std::string, std::vector<std::string>> rowwise_groups;
        std::map<std::string, std::vector<std::string>> rowwise_fit_groups;
        std::pair<double, double> feature_range = {-1.0, 1.0};

        // RNG seed for the NormalScore Monte-Carlo z-score generator.
        unsigned long seed = 0;

        // NS Monte-Carlo convergence params (pyemu defaults).
        double ns_tol = 1e-7;
        int ns_max_samples = 1000000;

        bool verbose = false;
    };

    explicit DsiEmulator(Config cfg);

    // Fit on training observation ensemble.
    //   training_obs: (n_real x n_obs), columns in cfg.obs_names order.
    void fit(const Eigen::MatrixXd& training_obs);

    // Predict observation ensembles from latent parameter values.
    //   latent_pvals: (n_real x ncomp), where ncomp == n_components().
    // Returns: (n_real x n_obs) in original (inverse-transformed)
    //   observation space, columns in cfg.obs_names order.
    Eigen::MatrixXd predict(const Eigen::MatrixXd& latent_pvals) const;

    // Project an observation ensemble into latent space.
    //   obs_ensemble: (n_real x n_obs), columns in cfg.obs_names order.
    // Returns: (n_real x ncomp) latent values such that
    //   predict(project_oe(oe)) is a truncation-projected reconstruction.
    Eigen::MatrixXd project_oe(const Eigen::MatrixXd& obs_ensemble) const;

    int n_components() const { return n_components_; }
    int n_obs() const { return static_cast<int>(cfg_.obs_names.size()); }
    bool is_fitted() const { return fitted_; }
    const std::vector<std::string>& obs_names() const { return cfg_.obs_names; }
    const Eigen::VectorXd& singular_values() const { return s_kept_; }
    const Eigen::VectorXd& ovals() const { return ovals_; }
    const Eigen::MatrixXd& pmat() const { return pmat_; }

    // Access to the NS transform inside the pipeline, so callers
    // (notably the golden-test loader) can pre-seed per-column state
    // before calling fit-with-pre-seeded path. Returns nullptr if no
    // NormalScore stage was registered.
    class NormalScoreTransform* normal_score_stage();

    // Convenience for tests / golden runs: skip the SVD-fitting on
    // training_obs but compute pmat / ovals from a pre-supplied set
    // of training data that has already been pushed through the
    // forward pipeline by the caller. Most users want fit().
    void fit_from_transformed(const Eigen::MatrixXd& transformed_train,
                              bool also_apply_pipeline_forward = false);

    // Mark the pipeline as fitted without a fit_apply call. Used when
    // a caller has externally seeded all transform state (notably the
    // golden test that pre-seeds NS state from a Python oracle).
    void mark_pipeline_fitted_externally() {
        pipeline_.mark_fitted_externally();
    }

    // Rowwise scaler accessors (testing / diagnostics).
    const DsiRowWiseScaler* train_rowwise_scaler() const {
        return train_rowwise_scaler_.get();
    }
    const DsiRowWiseScaler* truth_rowwise_scaler() const {
        return truth_rowwise_scaler_.get();
    }

private:
    void build_pipeline();
    DsiRowWiseScaler::GroupCols resolve_rowwise(
        const std::map<std::string, std::vector<std::string>>& spec) const;

    Config cfg_;
    DsiTransformPipeline pipeline_;
    std::unique_ptr<DsiRowWiseScaler> train_rowwise_scaler_;
    std::unique_ptr<DsiRowWiseScaler> truth_rowwise_scaler_;
    bool fitted_ = false;

    int n_components_ = 0;
    Eigen::VectorXd ovals_;     // (n_obs)  mean of transformed training data
    Eigen::MatrixXd v_kept_;    // (n_obs x ncomp) right singular vectors retained
    Eigen::VectorXd s_kept_;    // (ncomp)
    Eigen::MatrixXd pmat_;      // (n_obs x ncomp) = V_kept * diag(s_kept)
};

}  // namespace pestpp_dsi

#endif  // PESTPP_DSI_EMULATOR_H

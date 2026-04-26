#include "DsiEmulator.h"

#include <Eigen/SVD>
#include <cmath>
#include <stdexcept>

namespace pestpp_dsi {

DsiEmulator::DsiEmulator(Config cfg) : cfg_(std::move(cfg)) {
    if (cfg_.obs_names.empty()) {
        throw std::invalid_argument(
            "DsiEmulator: cfg.obs_names must not be empty");
    }
    if (!cfg_.obs_weights.empty()
        && cfg_.obs_weights.size() != cfg_.obs_names.size()) {
        throw std::invalid_argument(
            "DsiEmulator: obs_weights size mismatch with obs_names");
    }
    if (!cfg_.obs_truth.empty()
        && cfg_.obs_truth.size() != cfg_.obs_names.size()) {
        throw std::invalid_argument(
            "DsiEmulator: obs_truth size mismatch with obs_names");
    }
    build_pipeline();
}

void DsiEmulator::build_pipeline() {
    for (const auto& spec : cfg_.transforms) {
        std::vector<int> cols_idx;
        if (!spec.columns.empty()) {
            cols_idx = resolve_column_indices(spec.columns, cfg_.obs_names);
        }
        switch (spec.kind) {
            case TransformSpec::Kind::Identity:
                pipeline_.push_back(std::unique_ptr<DsiTransform>(
                    new IdentityTransform(std::move(cols_idx))));
                break;
            case TransformSpec::Kind::Log10:
                pipeline_.push_back(std::unique_ptr<DsiTransform>(
                    new Log10Transform(std::move(cols_idx))));
                break;
            case TransformSpec::Kind::NormalScore:
                pipeline_.push_back(std::unique_ptr<DsiTransform>(
                    new NormalScoreTransform(
                        std::move(cols_idx),
                        NSTailMode::Clip,
                        cfg_.ns_tol, cfg_.ns_max_samples, cfg_.seed)));
                break;
            case TransformSpec::Kind::NormalScoreLinear:
                pipeline_.push_back(std::unique_ptr<DsiTransform>(
                    new NormalScoreTransform(
                        std::move(cols_idx),
                        NSTailMode::Linear,
                        cfg_.ns_tol, cfg_.ns_max_samples, cfg_.seed)));
                break;
            case TransformSpec::Kind::NormalScoreQuad:
                pipeline_.push_back(std::unique_ptr<DsiTransform>(
                    new NormalScoreTransform(
                        std::move(cols_idx),
                        NSTailMode::Quad,
                        cfg_.ns_tol, cfg_.ns_max_samples, cfg_.seed)));
                break;
        }
    }
}

DsiRowWiseScaler::GroupCols DsiEmulator::resolve_rowwise(
    const std::map<std::string, std::vector<std::string>>& spec) const {
    DsiRowWiseScaler::GroupCols out;
    for (const auto& kv : spec) {
        out[kv.first] = resolve_column_indices(kv.second, cfg_.obs_names);
    }
    return out;
}

NormalScoreTransform* DsiEmulator::normal_score_stage() {
    // Pipeline does not expose its internals as raw pointers, so we
    // instead expose a hook by walking via dynamic_cast across the
    // staged transforms. This is only for golden-test pre-seeding
    // and is not used in production.
    //
    // Implementation note: we cannot scan because DsiTransformPipeline
    // hides its vector. Add a thin accessor instead.
    return pipeline_.find_normal_score_stage();
}

void DsiEmulator::fit_from_transformed(
    const Eigen::MatrixXd& X_t,
    bool also_apply_pipeline_forward) {
    Eigen::MatrixXd X = X_t;
    if (also_apply_pipeline_forward) pipeline_.apply(X);

    // Optional rowwise: fit on post-transform training matrix.
    if (!cfg_.rowwise_groups.empty()) {
        train_rowwise_scaler_.reset(new DsiRowWiseScaler(
            cfg_.feature_range,
            resolve_rowwise(cfg_.rowwise_groups),
            resolve_rowwise(cfg_.rowwise_fit_groups)));
        train_rowwise_scaler_->fit(X);
        train_rowwise_scaler_->apply(X);
    }

    ovals_ = X.colwise().mean();
    Eigen::MatrixXd dev = X.rowwise() - ovals_.transpose();
    const double scale = std::sqrt(static_cast<double>(X.rows() - 1));
    Eigen::MatrixXd Z = dev / scale;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        Z, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& s_full = svd.singularValues();
    const Eigen::MatrixXd& v_full = svd.matrixV();

    int n_keep = static_cast<int>(s_full.size());
    if (cfg_.energy_threshold < 1.0) {
        const double total = s_full.array().square().sum();
        if (total <= 0.0) {
            n_keep = 1;
        } else {
            double running = 0.0;
            const double cutoff = cfg_.energy_threshold * total;
            for (int i = 0; i < s_full.size(); ++i) {
                running += s_full(i) * s_full(i);
                if (running >= cutoff) { n_keep = i + 1; break; }
            }
        }
    }

    v_kept_ = v_full.leftCols(n_keep);
    s_kept_ = s_full.head(n_keep);
    pmat_ = v_kept_ * s_kept_.asDiagonal();
    n_components_ = n_keep;

    // Truth rowwise scaler: fit on a single-row truth vector after the
    // forward pipeline (plan §4.2 step 10). Used at predict time.
    if (!cfg_.rowwise_groups.empty()) {
        if (cfg_.obs_truth.empty()) {
            throw std::invalid_argument(
                "DsiEmulator: cfg.obs_truth required when rowwise_groups set");
        }
        Eigen::MatrixXd truth_row(1, cfg_.obs_names.size());
        for (Eigen::Index j = 0; j < truth_row.cols(); ++j) {
            truth_row(0, j) = cfg_.obs_truth[j];
        }
        // Apply forward transforms (without rowwise — that's what
        // the truth scaler will replace at predict time).
        pipeline_.apply(truth_row);
        truth_rowwise_scaler_.reset(new DsiRowWiseScaler(
            cfg_.feature_range,
            resolve_rowwise(cfg_.rowwise_groups),
            resolve_rowwise(cfg_.rowwise_fit_groups)));
        truth_rowwise_scaler_->fit(truth_row);
    }

    fitted_ = true;
}

void DsiEmulator::fit(const Eigen::MatrixXd& training_obs) {
    const Eigen::Index n_obs = static_cast<Eigen::Index>(cfg_.obs_names.size());
    if (training_obs.cols() != n_obs) {
        throw std::invalid_argument(
            "DsiEmulator::fit: training_obs.cols() != obs_names.size()");
    }
    if (training_obs.rows() < 2) {
        throw std::invalid_argument(
            "DsiEmulator::fit: need at least 2 realisations");
    }

    // Apply transforms (fits any per-stage state then forwards).
    Eigen::MatrixXd X = training_obs;
    pipeline_.fit_apply(X);

    // Then run the SVD/rowwise/truth-scaler half via the shared path.
    fit_from_transformed(X, /*also_apply_pipeline_forward=*/false);
}

Eigen::MatrixXd DsiEmulator::predict(
    const Eigen::MatrixXd& latent_pvals) const {
    if (!fitted_) {
        throw std::logic_error("DsiEmulator::predict called before fit");
    }
    if (latent_pvals.cols() != n_components_) {
        throw std::invalid_argument(
            "DsiEmulator::predict: latent_pvals.cols() != n_components()");
    }
    // sim_t (n_obs x n_real) = ovals[:, None] + pmat * latent.T
    Eigen::MatrixXd sim_t = (pmat_ * latent_pvals.transpose()).colwise()
                            + ovals_;
    Eigen::MatrixXd out = sim_t.transpose();        // (n_real x n_obs)

    // Truth rowwise inverse: scaler was fit on a single-row truth
    // vector, so its inverse broadcasts the truth's per-group
    // (t_min, t_max) to all realisations (plan §4.3 step 2).
    if (truth_rowwise_scaler_) {
        truth_rowwise_scaler_->inverse(out);
    }

    pipeline_.inverse(out);
    return out;
}

Eigen::MatrixXd DsiEmulator::project_oe(
    const Eigen::MatrixXd& obs_ensemble) const {
    if (!fitted_) {
        throw std::logic_error("DsiEmulator::project_oe called before fit");
    }
    if (obs_ensemble.cols() != static_cast<Eigen::Index>(cfg_.obs_names.size())) {
        throw std::invalid_argument(
            "DsiEmulator::project_oe: obs_ensemble.cols() != n_obs");
    }
    Eigen::MatrixXd oe_t = obs_ensemble;
    pipeline_.apply(oe_t);
    if (train_rowwise_scaler_) {
        // Apply the *training* scaler's forward to put oe in the same
        // space pmat lives in. Note: this only works exactly when
        // n_real(oe) == n_real(training), or n_real(oe) == 1 with the
        // training-row[0] params broadcast — which is the typical
        // pattern for projecting individual realisations.
        train_rowwise_scaler_->apply(oe_t);
    }
    Eigen::MatrixXd resid = oe_t.rowwise() - ovals_.transpose();
    // pinv(pmat) = pinv(V * diag(s)) = diag(1/s) * V^T  since V has
    // orthonormal columns. So latent = resid * V * diag(1/s).
    Eigen::VectorXd inv_s = s_kept_.array().inverse();
    return (resid * v_kept_) * inv_s.asDiagonal();
}

}  // namespace pestpp_dsi

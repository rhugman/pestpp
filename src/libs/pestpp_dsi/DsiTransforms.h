// DsiTransforms.h
//
// Per-column data-space transforms applied to the training observation
// ensemble before the DSI compact SVD. Mirrors the pyemu transformer
// chain in dependencies/pyemu/pyemu/emulators/transformers.py.
//
// Phase 1 ships Identity and Log10. NormalScore is added in Phase 2.

#ifndef PESTPP_DSI_TRANSFORMS_H
#define PESTPP_DSI_TRANSFORMS_H

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace pestpp_dsi {

// Abstract base. Each transform owns a list of column indices that
// identifies which columns of the (n_real x n_obs) data matrix it
// acts on. An empty cols_idx list means "all columns".
class DsiTransform {
public:
    explicit DsiTransform(std::vector<int> cols_idx);
    virtual ~DsiTransform();

    // Fit any internal state from the training matrix. Default: no-op.
    virtual void fit(const Eigen::MatrixXd& X);

    // Forward transform (in place).
    virtual void apply(Eigen::MatrixXd& X) const = 0;

    // Inverse transform (in place).
    virtual void inverse(Eigen::MatrixXd& X) const = 0;

    const std::vector<int>& cols_idx() const { return cols_idx_; }

protected:
    // Resolve "empty == all" once at fit/apply time.
    std::vector<int> resolved_cols(int n_cols) const;

    std::vector<int> cols_idx_;
};

// No-op transform.
class IdentityTransform : public DsiTransform {
public:
    explicit IdentityTransform(std::vector<int> cols_idx = {});
    void apply(Eigen::MatrixXd&) const override {}
    void inverse(Eigen::MatrixXd&) const override {}
};

// Stateless log10 transform.
//
// Precondition: the columns this transform acts on must be strictly
// positive on the training data. The DSI workflow that calls this
// transform is responsible for not applying log10 to columns that
// can hit zero or negative values (matches pyemu behaviour, which
// also does not silently shift).
class Log10Transform : public DsiTransform {
public:
    explicit Log10Transform(std::vector<int> cols_idx = {});
    void apply(Eigen::MatrixXd& X) const override;
    void inverse(Eigen::MatrixXd& X) const override;
};

// Tail-extrapolation policy for NormalScoreTransform on out-of-range inputs:
//   Clip   — clamp to the boundary value (no extrapolation).
//   Linear — extend with the slope between the last two training points.
//            (This is what pyemu mis-labels as "quadratic_extrapolation=True".)
//   Quad   — Lagrange-quadratic curve through the last three training points
//            (true quadratic; smooth tail continuation).
enum class NSTailMode { Clip, Linear, Quad };

// Normal-score transform — port of
// pyemu.emulators.transformers.NormalScoreTransformer. Per-column
// state is `originals` (sorted, smoothed, monotone-enforced training
// values) and `z_scores` (sample-then-sort standard normals). Forward
// transform is linear interpolation `originals -> z_scores`; inverse
// is `z_scores -> originals`. Out-of-range inputs follow `tail_mode_`.
//
// Note on naming: pyemu's `quadratic_extrapolation` flag is a misnomer
// — its True branch is LINEAR (boundary slope), not quadratic. This
// class exposes a 3-state mode (Clip / Linear / Quad) where Quad is
// the true Lagrange-quadratic continuation through the last 3 points.
//
// Reproducibility: production NS uses the C++ RNG seeded from cfg.seed;
// for golden tests against pyemu, callers can pre-seed per-column
// state via set_state() to bypass any std::normal_distribution cross-
// compiler portability issues (plan §5.2).
class NormalScoreTransform : public DsiTransform {
public:
    explicit NormalScoreTransform(std::vector<int> cols_idx = {},
                                  NSTailMode tail_mode = NSTailMode::Quad,
                                  double tol = 1e-7,
                                  int max_samples = 1000000,
                                  unsigned long seed = 0);

    void fit(const Eigen::MatrixXd& X) override;
    void apply(Eigen::MatrixXd& X) const override;
    void inverse(Eigen::MatrixXd& X) const override;

    // Pre-seed the state for a single active column (col_pos = index
    // into cols_idx_, or 0..n_obs-1 when cols_idx_ is empty/all). Both
    // arrays must be ascending and the same length. Used by the
    // golden-test loader to inject pyemu-trained state.
    void set_state(int col_pos,
                   Eigen::VectorXd z_scores,
                   Eigen::VectorXd originals);

    bool is_fitted() const { return fitted_; }
    int n_active_cols() const { return static_cast<int>(state_.size()); }
    const Eigen::VectorXd& z_scores(int col_pos) const;
    const Eigen::VectorXd& originals(int col_pos) const;

    // Public for unit testing — mirrors transformers.py:638-674.
    static Eigen::VectorXd moving_average_with_endpoints(
        const Eigen::VectorXd& sorted_y);

    // Public for unit testing — mirrors transformers.py:613-636.
    Eigen::VectorXd sample_z_scores(int nreal);

private:
    struct ColState {
        Eigen::VectorXd z_scores;   // ascending
        Eigen::VectorXd originals;  // ascending, smoothed, monotone
        bool seeded = false;
    };

    void ensure_state_size();
    int active_col_count(int n_cols) const;
    void apply_one(Eigen::MatrixXd& X, int j, const ColState& st) const;
    void inverse_one(Eigen::MatrixXd& X, int j, const ColState& st) const;

    NSTailMode tail_mode_;
    double tol_;
    int max_samples_;
    bool fitted_ = false;
    mutable std::mt19937_64 rng_;
    std::vector<ColState> state_;   // length == active column count
};

// 1-D linear interpolation, np.interp-compatible for monotonically
// strictly increasing xp. For values outside [xp.front(), xp.back()]
// returns the corresponding endpoint of fp (clamping). For exactly-
// equal x at a knot, returns the corresponding fp at that knot.
double interp1d(double x,
                const Eigen::VectorXd& xp,
                const Eigen::VectorXd& fp);

// Helper: resolve a list of obs-name strings to integer column indices
// against a column-name vector. Throws std::invalid_argument on
// missing names. Used at pipeline-construction time.
std::vector<int> resolve_column_indices(
    const std::vector<std::string>& names,
    const std::vector<std::string>& obs_names);

}  // namespace pestpp_dsi

#endif  // PESTPP_DSI_TRANSFORMS_H

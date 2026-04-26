#include "DsiTransforms.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

namespace pestpp_dsi {

DsiTransform::DsiTransform(std::vector<int> cols_idx)
    : cols_idx_(std::move(cols_idx)) {}

DsiTransform::~DsiTransform() = default;

void DsiTransform::fit(const Eigen::MatrixXd&) {}

std::vector<int> DsiTransform::resolved_cols(int n_cols) const {
    if (!cols_idx_.empty()) return cols_idx_;
    std::vector<int> all(n_cols);
    for (int i = 0; i < n_cols; ++i) all[i] = i;
    return all;
}

IdentityTransform::IdentityTransform(std::vector<int> cols_idx)
    : DsiTransform(std::move(cols_idx)) {}

Log10Transform::Log10Transform(std::vector<int> cols_idx)
    : DsiTransform(std::move(cols_idx)) {}

void Log10Transform::apply(Eigen::MatrixXd& X) const {
    const auto cols = resolved_cols(static_cast<int>(X.cols()));
    for (int j : cols) {
        X.col(j) = X.col(j).array().log10().matrix();
    }
}

void Log10Transform::inverse(Eigen::MatrixXd& X) const {
    const auto cols = resolved_cols(static_cast<int>(X.cols()));
    for (int j : cols) {
        // 10^x = exp(x * ln(10)). Eigen Array supports .pow(scalar) as
        // x.pow(s); but for "10 to the x" we need pow(10, x). Use
        // scalar exp form for portability with Eigen 3.3.
        X.col(j) = (X.col(j).array() * std::log(10.0)).exp().matrix();
    }
}

std::vector<int> resolve_column_indices(
    const std::vector<std::string>& names,
    const std::vector<std::string>& obs_names) {
    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx.reserve(obs_names.size());
    for (int i = 0; i < static_cast<int>(obs_names.size()); ++i) {
        name_to_idx.emplace(obs_names[i], i);
    }
    std::vector<int> out;
    out.reserve(names.size());
    for (const auto& nm : names) {
        auto it = name_to_idx.find(nm);
        if (it == name_to_idx.end()) {
            throw std::invalid_argument(
                "DsiTransform: obs name '" + nm + "' not in obs_names");
        }
        out.push_back(it->second);
    }
    return out;
}

// ---- np.interp port -------------------------------------------------------
//
// np.interp(x, xp, fp) for monotonically increasing xp:
//   - clamp to fp.front() if x < xp.front()
//   - clamp to fp.back()  if x > xp.back()
//   - linear interpolation otherwise
// Ties: numpy returns the "first match" — but we guarantee strict
// monotonicity in the NS smoothing step (the +=1e-16 tied-rank
// protection at transformers.py:670-672), so ties never appear in
// practice.

double interp1d(double x,
                const Eigen::VectorXd& xp,
                const Eigen::VectorXd& fp) {
    if (xp.size() != fp.size() || xp.size() == 0) {
        throw std::invalid_argument("interp1d: xp/fp size mismatch or empty");
    }
    const Eigen::Index n = xp.size();
    if (x <= xp(0)) return fp(0);
    if (x >= xp(n - 1)) return fp(n - 1);
    // lower_bound: first iter such that *iter >= x. Since xp is strictly
    // monotonic and x is strictly inside the range, this gives an
    // index in [1, n-1].
    const double* xp_data = xp.data();
    const double* hi = std::lower_bound(xp_data, xp_data + n, x);
    const Eigen::Index i = static_cast<Eigen::Index>(hi - xp_data);
    if (i == 0) return fp(0);                 // defensive (shouldn't hit)
    if (i >= n) return fp(n - 1);             // defensive
    const double x0 = xp(i - 1), x1 = xp(i);
    const double f0 = fp(i - 1), f1 = fp(i);
    const double t = (x - x0) / (x1 - x0);
    return f0 + t * (f1 - f0);
}

// ---- NormalScoreTransform -------------------------------------------------
//
// Faithful port of pyemu.emulators.transformers.NormalScoreTransformer.
// Each port target referenced below cites the line range in the upstream
// transformers.py source.

NormalScoreTransform::NormalScoreTransform(std::vector<int> cols_idx,
                                           bool quadratic_extrapolation,
                                           double tol,
                                           int max_samples,
                                           unsigned long seed)
    : DsiTransform(std::move(cols_idx)),
      quadratic_extrapolation_(quadratic_extrapolation),
      tol_(tol),
      max_samples_(max_samples),
      rng_(seed) {}

int NormalScoreTransform::active_col_count(int n_cols) const {
    return cols_idx_.empty() ? n_cols : static_cast<int>(cols_idx_.size());
}

void NormalScoreTransform::ensure_state_size() {
    // Caller has resolved how many active cols there are by passing it
    // through the public-facing seed_state path or by calling fit()
    // with a real matrix. This is an internal guard for set_state.
}

void NormalScoreTransform::set_state(int col_pos,
                                     Eigen::VectorXd z_scores,
                                     Eigen::VectorXd originals) {
    if (z_scores.size() != originals.size()) {
        throw std::invalid_argument(
            "NormalScoreTransform::set_state: z_scores/originals size mismatch");
    }
    if (col_pos < 0) {
        throw std::invalid_argument(
            "NormalScoreTransform::set_state: col_pos must be >= 0");
    }
    if (static_cast<int>(state_.size()) <= col_pos) {
        state_.resize(col_pos + 1);
    }
    state_[col_pos].z_scores = std::move(z_scores);
    state_[col_pos].originals = std::move(originals);
    state_[col_pos].seeded = true;
    fitted_ = true;
}

const Eigen::VectorXd& NormalScoreTransform::z_scores(int col_pos) const {
    if (col_pos < 0 || col_pos >= static_cast<int>(state_.size())) {
        throw std::out_of_range("NormalScoreTransform::z_scores: col_pos out of range");
    }
    return state_[col_pos].z_scores;
}
const Eigen::VectorXd& NormalScoreTransform::originals(int col_pos) const {
    if (col_pos < 0 || col_pos >= static_cast<int>(state_.size())) {
        throw std::out_of_range("NormalScoreTransform::originals: col_pos out of range");
    }
    return state_[col_pos].originals;
}

// transformers.py:638-674 — moving average with endpoint preservation
// and monotonic-tied-rank protection.
Eigen::VectorXd NormalScoreTransform::moving_average_with_endpoints(
    const Eigen::VectorXd& y_values) {
    const Eigen::Index n = y_values.size();
    if (n == 0) return Eigen::VectorXd();

    // Window-size step function (transformers.py:640-646). Faithful
    // port of the literal thresholds; preserves the discontinuity at
    // n_train == 41/91/201 (plan §5.3 port target #2).
    int window_size = 3;
    if (n > 40) window_size = 5;
    if (n > 90) window_size = 7;
    if (n > 200) window_size = 9;
    if (window_size % 2 == 0) {
        throw std::logic_error("window_size must be odd");
    }
    const int half_window = window_size / 2;

    Eigen::VectorXd smoothed = Eigen::VectorXd::Zero(n);

    // Start half-window (transformers.py:654-655).
    // for i in 0..half_window-1: mean of y[:i + half_window + 1].
    for (int i = 0; i < half_window && i < static_cast<int>(n); ++i) {
        const Eigen::Index hi = std::min<Eigen::Index>(
            i + half_window + 1, n);
        smoothed(i) = y_values.head(hi).mean();
    }

    // End half-window (transformers.py:658-659).
    // for i in 1..half_window: smoothed[-i] = mean(y[-(i+half_window):]).
    for (int i = 1; i <= half_window && (n - i) >= 0; ++i) {
        const Eigen::Index take = std::min<Eigen::Index>(i + half_window, n);
        smoothed(n - i) = y_values.tail(take).mean();
    }

    // Middle (transformers.py:662-663).
    for (Eigen::Index i = half_window; i + half_window < n; ++i) {
        smoothed(i) = y_values.segment(i - half_window,
                                       2 * half_window + 1).mean();
    }

    // Endpoint overwrite (transformers.py:665-667).
    // This is *after* the loops — the loops produce some smoothed
    // value for index 0 and index n-1, and those values are then
    // unconditionally overwritten with the originals. Plan §5.3 port
    // target #3 explicitly calls this out.
    if (n > 0) {
        smoothed(0) = y_values(0);
        smoothed(n - 1) = y_values(n - 1);
    }

    // Tied-rank monotonic enforcement (transformers.py:670-672).
    // Literal `1e-16`, NOT std::numeric_limits<double>::epsilon().
    // Plan §5.3 port target #4.
    for (Eigen::Index i = 1; i < n; ++i) {
        if (smoothed(i) <= smoothed(i - 1)) {
            smoothed(i) = smoothed(i - 1) + 1e-16;
        }
    }

    return smoothed;
}

// transformers.py:613-636 — Monte-Carlo sample-then-sort z-score
// generator. Anti-symmetric closed form: rval[k] = -rval[n-1-k] when
// n even; the middle value is exactly 0 when n odd.
Eigen::VectorXd NormalScoreTransform::sample_z_scores(int nreal) {
    if (nreal <= 0) {
        throw std::invalid_argument(
            "NormalScoreTransform::sample_z_scores: nreal must be positive");
    }
    Eigen::VectorXd rval = Eigen::VectorXd::Zero(nreal);
    const int numsort = (nreal % 2 == 0) ? (nreal + 1) / 2 : nreal / 2;
    int nsamp = 0;
    std::normal_distribution<double> N(0.0, 1.0);
    Eigen::VectorXd previous_mean = Eigen::VectorXd::Zero(numsort);

    while (nsamp < max_samples_) {
        ++nsamp;
        Eigen::VectorXd work1(nreal);
        for (int i = 0; i < nreal; ++i) work1(i) = N(rng_);
        std::sort(work1.data(), work1.data() + nreal);

        if (nsamp > 1) {
            previous_mean = rval.head(numsort) / static_cast<double>(nsamp - 1);
            rval.head(numsort) += work1.head(numsort);
            Eigen::VectorXd current_mean
                = rval.head(numsort) / static_cast<double>(nsamp);
            const double max_diff = (current_mean - previous_mean)
                                        .array().abs().maxCoeff();
            if (max_diff <= tol_) break;
        } else {
            rval.head(numsort) = work1.head(numsort);
        }
    }
    rval.head(numsort) /= static_cast<double>(nsamp);

    if (nreal % 2 == 0) {
        // rval[numsort:] = -rval[:numsort][::-1].
        for (int k = 0; k < numsort; ++k) {
            rval(numsort + k) = -rval(numsort - 1 - k);
        }
    } else {
        // rval[numsort:] = concat(([-rval[numsort]], -rval[:numsort][::-1])).
        // rval(numsort) is the middle slot; pyemu negates the
        // *current* value at rval[numsort] (which is 0 because we
        // never wrote to it), so the middle stays 0.
        const double mid = rval(numsort);
        rval(numsort) = -mid;
        for (int k = 0; k < numsort; ++k) {
            rval(numsort + 1 + k) = -rval(numsort - 1 - k);
        }
    }
    return rval;
}

void NormalScoreTransform::fit(const Eigen::MatrixXd& X) {
    const std::vector<int> cols = resolved_cols(static_cast<int>(X.cols()));
    state_.assign(cols.size(), ColState{});
    for (std::size_t k = 0; k < cols.size(); ++k) {
        const int j = cols[k];
        Eigen::VectorXd vals = X.col(j);
        std::sort(vals.data(), vals.data() + vals.size());
        Eigen::VectorXd smoothed = moving_average_with_endpoints(vals);
        Eigen::VectorXd zs = sample_z_scores(static_cast<int>(smoothed.size()));
        state_[k].z_scores = std::move(zs);
        state_[k].originals = std::move(smoothed);
        state_[k].seeded = true;
    }
    fitted_ = true;
}

void NormalScoreTransform::apply_one(
    Eigen::MatrixXd& X, int j, const ColState& st) const {
    const Eigen::VectorXd& orig = st.originals;
    const Eigen::VectorXd& zs = st.z_scores;
    const double min_orig = orig(0);
    const double max_orig = orig(orig.size() - 1);
    const double min_z = zs(0);
    const double max_z = zs(zs.size() - 1);
    for (Eigen::Index r = 0; r < X.rows(); ++r) {
        const double v = X(r, j);
        if (v >= min_orig && v <= max_orig) {
            X(r, j) = interp1d(v, orig, zs);
        } else if (v < min_orig) {
            if (quadratic_extrapolation_) {
                const double slope = (zs(1) - zs(0)) / (orig(1) - orig(0));
                X(r, j) = min_z + slope * (v - min_orig);
            } else {
                X(r, j) = min_z;
            }
        } else { // v > max_orig
            if (quadratic_extrapolation_) {
                const Eigen::Index n = zs.size();
                const double slope = (zs(n - 1) - zs(n - 2))
                                     / (orig(n - 1) - orig(n - 2));
                X(r, j) = max_z + slope * (v - max_orig);
            } else {
                X(r, j) = max_z;
            }
        }
    }
}

void NormalScoreTransform::inverse_one(
    Eigen::MatrixXd& X, int j, const ColState& st) const {
    const Eigen::VectorXd& orig = st.originals;
    const Eigen::VectorXd& zs = st.z_scores;
    const double min_orig = orig(0);
    const double max_orig = orig(orig.size() - 1);
    const double min_z = zs(0);
    const double max_z = zs(zs.size() - 1);
    for (Eigen::Index r = 0; r < X.rows(); ++r) {
        const double v = X(r, j);
        if (v >= min_z && v <= max_z) {
            X(r, j) = interp1d(v, zs, orig);
        } else if (v < min_z) {
            if (quadratic_extrapolation_) {
                const double slope = (orig(1) - orig(0)) / (zs(1) - zs(0));
                const double intercept = orig(0) - slope * zs(0);
                X(r, j) = slope * v + intercept;
            } else {
                X(r, j) = min_orig;
            }
        } else { // v > max_z
            if (quadratic_extrapolation_) {
                const Eigen::Index n = zs.size();
                const double slope = (orig(n - 1) - orig(n - 2))
                                     / (zs(n - 1) - zs(n - 2));
                const double intercept = orig(n - 1) - slope * zs(n - 1);
                X(r, j) = slope * v + intercept;
            } else {
                X(r, j) = max_orig;
            }
        }
    }
}

void NormalScoreTransform::apply(Eigen::MatrixXd& X) const {
    if (!fitted_) {
        throw std::logic_error("NormalScoreTransform::apply called before fit");
    }
    const std::vector<int> cols = resolved_cols(static_cast<int>(X.cols()));
    if (cols.size() != state_.size()) {
        throw std::logic_error(
            "NormalScoreTransform::apply: column count != fit state size");
    }
    for (std::size_t k = 0; k < cols.size(); ++k) {
        if (!state_[k].seeded) continue;
        apply_one(X, cols[k], state_[k]);
    }
}

void NormalScoreTransform::inverse(Eigen::MatrixXd& X) const {
    if (!fitted_) {
        throw std::logic_error("NormalScoreTransform::inverse called before fit");
    }
    const std::vector<int> cols = resolved_cols(static_cast<int>(X.cols()));
    if (cols.size() != state_.size()) {
        throw std::logic_error(
            "NormalScoreTransform::inverse: column count != fit state size");
    }
    for (std::size_t k = 0; k < cols.size(); ++k) {
        if (!state_[k].seeded) continue;
        inverse_one(X, cols[k], state_[k]);
    }
}

}  // namespace pestpp_dsi

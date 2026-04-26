// test_normal_score.cpp
//
// Plan §10.1 NS coverage (round-trip identity):
//   inverse(transform(x)) ~= x to 1e-12 element-wise on the *training*
//   data (same data the transform was fit on). Out-of-range values
//   are not part of round-trip identity (they get clamped under the
//   default `quadratic_extrapolation=False` mode), so we don't include
//   them here.
//
// Distributions covered: uniform, log-normal, multi-modal, all-equal
// (degenerate). Sizes covered: small (N=4), and N just above the
// MA-window step thresholds (40, 90, 200) — plan §5.3 port target #2.

#include "DsiTransforms.h"
#include "test_helpers.h"

#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace {

bool round_trip_ok(const Eigen::VectorXd& col, double tol,
                   bool quad_extrap = false) {
    using pestpp_dsi::NormalScoreTransform;
    Eigen::MatrixXd X(col.size(), 1);
    X.col(0) = col;
    // tol/max_samples are the Monte-Carlo z-score convergence params.
    // Relaxed here because round-trip identity only requires a
    // well-defined (z_scores, originals) pair, not pyemu-tight
    // convergence. Production defaults (1e-7, 1e6) are unchanged.
    NormalScoreTransform ns({}, quad_extrap, /*tol=*/1e-3,
                            /*max_samples=*/2000);
    ns.fit(X);

    Eigen::MatrixXd Z = X;
    ns.apply(Z);
    Eigen::MatrixXd back = Z;
    ns.inverse(back);

    const double max_abs = (back - X).array().abs().maxCoeff();
    if (max_abs > tol) {
        std::cerr << "  round_trip: max|back - X| = " << max_abs
                  << " > " << tol << " (n=" << col.size() << ")\n";
        return false;
    }
    return true;
}

Eigen::VectorXd uniform(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = U(rng);
    return v;
}

Eigen::VectorXd lognormal(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::lognormal_distribution<double> LN(0.0, 1.0);
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = LN(rng);
    return v;
}

Eigen::VectorXd bimodal(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> N1(-3.0, 0.5);
    std::normal_distribution<double> N2(3.0, 0.5);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = U(rng) < 0.5 ? N1(rng) : N2(rng);
    return v;
}

}  // namespace

int main() {
    using pestpp_dsi::NormalScoreTransform;

    // --- Round-trip identity across distributions --------------------
    DSI_EXPECT(round_trip_ok(uniform(50, 1u), 1e-12));
    DSI_EXPECT(round_trip_ok(lognormal(50, 2u), 1e-9));   // log-normal
        // can have large values; absolute tol relaxed slightly
    DSI_EXPECT(round_trip_ok(bimodal(50, 3u), 1e-12));

    // --- Boundary sizes around MA-window step thresholds -------------
    // Plan §5.3 port target #2: thresholds 40, 90, 200 (window jumps
    // 3->5->7->9 at strict-greater-than). Test n=4 (small), n=41 (just
    // past 40), n=91, n=201.
    for (int n : {4, 41, 91, 201}) {
        DSI_EXPECT(round_trip_ok(uniform(n, 7u + n), 1e-12));
    }

    // --- Degenerate: all-equal training values -----------------------
    // After np.sort + smoothing, the +=1e-16 monotonicity protection
    // kicks in (transformers.py:670-672). Round-trip should still hold
    // approximately on the *training* data: the transform maps each
    // sorted index to its z-score; the inverse maps back to the
    // (now strictly-monotone) smoothed originals. Since the originals
    // start all-equal, the smoothed ones are all within 1e-16 ulps,
    // and the inverse picks them up by interpolation — so round-trip
    // returns the smoothed values, NOT the original constants. We
    // therefore relax the test: the deviation must be <= ~ N * 1e-16.
    {
        const int n = 30;
        Eigen::VectorXd col = Eigen::VectorXd::Constant(n, 7.5);
        Eigen::MatrixXd X(n, 1); X.col(0) = col;
        NormalScoreTransform ns({}, /*quad=*/false, 1e-3, 2000);
        ns.fit(X);
        Eigen::MatrixXd Z = X;
        ns.apply(Z);
        ns.inverse(Z);
        const double max_dev = (Z - X).array().abs().maxCoeff();
        DSI_EXPECT(max_dev <= n * 1e-15);
    }

    // --- Tiny sample (N=4) just to make sure we don't blow up --------
    DSI_EXPECT(round_trip_ok(uniform(4, 99u), 1e-12));

    // --- _moving_average_with_endpoints sanity ----------------------
    // After smoothing, originals must be strictly monotone non-decreasing
    // by construction (the +=1e-16 fix). Spot-check on uniform(91)
    // (which uses window_size=7).
    {
        Eigen::VectorXd vals = uniform(91, 11u);
        std::sort(vals.data(), vals.data() + vals.size());
        Eigen::VectorXd smoothed
            = NormalScoreTransform::moving_average_with_endpoints(vals);
        DSI_EXPECT(smoothed.size() == vals.size());
        DSI_EXPECT_NEAR(smoothed(0), vals(0), 0.0);
        DSI_EXPECT_NEAR(smoothed(smoothed.size() - 1),
                        vals(vals.size() - 1), 0.0);
        for (Eigen::Index i = 1; i < smoothed.size(); ++i) {
            DSI_EXPECT(smoothed(i) > smoothed(i - 1));
        }
    }

    // --- Tail-extrapolation modes: Clip / Linear / Quad --------------
    // Three distinct out-of-range behaviors. Confirm they give different
    // answers and that round-trip through the same mode preserves the
    // out-of-range input for Linear and Quad (Clip clamps to the
    // boundary so it cannot round-trip).
    {
        Eigen::VectorXd col = uniform(50, 5u);
        Eigen::MatrixXd X(col.size(), 1); X.col(0) = col;

        NormalScoreTransform ns_clip   ({}, NSTailMode::Clip,   1e-3, 2000);
        NormalScoreTransform ns_linear ({}, NSTailMode::Linear, 1e-3, 2000);
        NormalScoreTransform ns_quad   ({}, NSTailMode::Quad,   1e-3, 2000);
        ns_clip.fit(X);
        ns_linear.fit(X);
        ns_quad.fit(X);

        // Value clearly above the training max.
        Eigen::MatrixXd Y(1, 1); Y(0, 0) = col.maxCoeff() + 5.0;
        Eigen::MatrixXd Yc = Y, Yl = Y, Yq = Y;
        ns_clip.apply(Yc);
        ns_linear.apply(Yl);
        ns_quad.apply(Yq);
        // All three should disagree on out-of-range input.
        DSI_EXPECT(std::abs(Yl(0, 0) - Yc(0, 0)) > 0.1);
        DSI_EXPECT(std::abs(Yq(0, 0) - Yc(0, 0)) > 0.1);
        DSI_EXPECT(std::abs(Yq(0, 0) - Yl(0, 0)) > 1e-6);

        // Round-trip preserves the out-of-range value for Linear and
        // Quad (the inverse undoes the same extrapolation rule).
        Eigen::MatrixXd Yl_inv = Yl, Yq_inv = Yq;
        ns_linear.inverse(Yl_inv);
        ns_quad.inverse(Yq_inv);
        DSI_EXPECT_NEAR(Yl_inv(0, 0), Y(0, 0), 1e-10);
        DSI_EXPECT_NEAR(Yq_inv(0, 0), Y(0, 0), 1e-10);

        // Clip cannot round-trip — the forward sends Y(0,0) to max_z,
        // and the inverse sends it back to max_orig (= original max,
        // not the input's max+5).
        Eigen::MatrixXd Yc_inv = Yc;
        ns_clip.inverse(Yc_inv);
        DSI_EXPECT(std::abs(Yc_inv(0, 0) - Y(0, 0)) > 1.0);
    }

    return EXIT_SUCCESS;
}

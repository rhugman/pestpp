// test_normal_score_clustered_boundary.cpp
//
// B13_ns_extrap_plan.md Test C — clustered-boundary unit test.
//
// Reproduces the failure mode that drives the ies_1 surrogate
// explosion: a 100-real training column where 99 reals are 0.0 and
// 1 real is 1.0. After np.sort + moving_average_with_endpoints, the
// upper-end pair (orig(n-2), orig(n-1)) is protected by the literal
// 1e-16 tied-rank floor at DsiTransforms.cpp:234. The forward
// boundary slope (zs(n-1)-zs(n-2)) / (orig(n-1)-orig(n-2)) is then
// of order 1e+16 in native obs units, and any forward-transformed
// out-of-range value produces a z-score of similar magnitude. The
// safe-boundary-slope helper added in B14 walks inward to the first
// meaningful pair and the z-output cap clamps to z_cap = ~10 for
// n=100.
//
// Pass criterion: |z| <= 12.0 for all out-of-range inputs (z_cap is
// |min_z| + |max_z| + 5.0 ~ 2*2.5 + 5 = 10; allow 2 units slack for
// future ensemble sizes).

#include "DsiTransforms.h"
#include "test_helpers.h"

#include <cmath>
#include <iostream>

int main() {
    using pestpp_dsi::NormalScoreTransform;
    using pestpp_dsi::NSTailMode;

    // 100x1 training matrix: 99 reals at 0.0, 1 real at 1.0.
    const int n = 100;
    Eigen::MatrixXd train(n, 1);
    for (int i = 0; i < n - 1; ++i) train(i, 0) = 0.0;
    train(n - 1, 0) = 1.0;

    NormalScoreTransform ns({}, NSTailMode::Linear,
                            /*tol=*/1e-3, /*max_samples=*/2000);
    ns.fit(train);

    // Forward-transform a small in-range value, a moderately
    // out-of-range value, and a far-out-of-range value.
    Eigen::MatrixXd Y(3, 1);
    Y(0, 0) = 0.5;     // in-range (between 0 and 1)
    Y(1, 0) = 2.0;     // upper out-of-range
    Y(2, 0) = 100.0;   // far upper out-of-range

    Eigen::MatrixXd Z = Y;
    ns.apply(Z);

    std::cerr << "  clustered-boundary forward z-scores: "
              << Z(0, 0) << ", " << Z(1, 0) << ", " << Z(2, 0)
              << "\n";

    // All three forward outputs must be bounded by ~12 (the z_cap of
    // |min_z|+|max_z|+5 for an n=100 ensemble where max-z ~ 2.5).
    DSI_EXPECT(std::abs(Z(0, 0)) <= 12.0);
    DSI_EXPECT(std::abs(Z(1, 0)) <= 12.0);
    DSI_EXPECT(std::abs(Z(2, 0)) <= 12.0);

    // Without the fix, Z(1,0) and Z(2,0) would be ~1e+15. Sanity-
    // check the order of magnitude is now reasonable for normal-
    // score outputs (single-digit z-scores in the cap region).
    DSI_EXPECT(std::isfinite(Z(0, 0)));
    DSI_EXPECT(std::isfinite(Z(1, 0)));
    DSI_EXPECT(std::isfinite(Z(2, 0)));

    return EXIT_SUCCESS;
}

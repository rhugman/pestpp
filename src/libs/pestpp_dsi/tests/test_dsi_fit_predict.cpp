// test_dsi_fit_predict.cpp
//
// 50 reals × 100 obs random Gaussian training, no transforms,
// energy_threshold=1.0. predict(project_oe(training)) must reconstruct
// the training data to ~1e-10 (no truncation, no transforms => clean
// round-trip).

#include "DsiEmulator.h"
#include "test_helpers.h"

#include <random>
#include <string>
#include <vector>

int main() {
    using pestpp_dsi::DsiEmulator;

    constexpr int n_real = 50;
    constexpr int n_obs = 100;

    std::mt19937 rng(42u);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd X(n_real, n_obs);
    for (int i = 0; i < n_real; ++i)
        for (int j = 0; j < n_obs; ++j) X(i, j) = dist(rng);

    DsiEmulator::Config cfg;
    cfg.obs_names.reserve(n_obs);
    for (int j = 0; j < n_obs; ++j)
        cfg.obs_names.push_back("o" + std::to_string(j));
    cfg.energy_threshold = 1.0;       // keep all components

    DsiEmulator dsi(cfg);
    dsi.fit(X);

    DSI_EXPECT(dsi.is_fitted());
    // At energy_threshold=1.0 we keep every singular value JacobiSVD
    // returns (matches pyemu's "skip truncation" branch). Compact SVD
    // of (50 x 100) returns min(50, 100) = 50 components; the 50th
    // is numerically near-zero (centered matrix has rank N-1=49) but
    // not pruned without an explicit threshold.
    DSI_EXPECT(dsi.n_components() == 50);

    // Round-trip: project then predict.
    Eigen::MatrixXd latent = dsi.project_oe(X);
    DSI_EXPECT(latent.rows() == n_real);
    DSI_EXPECT(latent.cols() == dsi.n_components());

    Eigen::MatrixXd recon = dsi.predict(latent);
    DSI_EXPECT(recon.rows() == n_real);
    DSI_EXPECT(recon.cols() == n_obs);

    DSI_EXPECT_MATRIX_CLOSE(recon, X, 1e-9);

    return EXIT_SUCCESS;
}

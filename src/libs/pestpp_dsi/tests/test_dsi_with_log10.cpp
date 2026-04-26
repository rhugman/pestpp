// test_dsi_with_log10.cpp
//
// Phase 1 sanity test that the transform pipeline interacts correctly
// with fit/predict/project_oe. Train on positive log-normal data with
// a Log10 transform, then round-trip through project_oe -> predict.
// With energy_threshold=1.0 the result must match the input.

#include "DsiEmulator.h"
#include "test_helpers.h"

#include <random>
#include <string>
#include <vector>

int main() {
    using pestpp_dsi::DsiEmulator;

    constexpr int n_real = 40;
    constexpr int n_obs = 30;

    std::mt19937 rng(7u);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd X(n_real, n_obs);
    // Log-normal: exp of standard normal so all values are strictly
    // positive (Log10 transform precondition).
    for (int i = 0; i < n_real; ++i)
        for (int j = 0; j < n_obs; ++j) X(i, j) = std::exp(dist(rng));

    DsiEmulator::Config cfg;
    cfg.obs_names.reserve(n_obs);
    for (int j = 0; j < n_obs; ++j)
        cfg.obs_names.push_back("o" + std::to_string(j));
    cfg.energy_threshold = 1.0;
    DsiEmulator::TransformSpec spec;
    spec.kind = DsiEmulator::TransformSpec::Kind::Log10;     // all columns
    cfg.transforms.push_back(spec);

    DsiEmulator dsi(cfg);
    dsi.fit(X);

    Eigen::MatrixXd latent = dsi.project_oe(X);
    Eigen::MatrixXd recon = dsi.predict(latent);

    // Element-wise relative agreement: log-normal values can span many
    // orders of magnitude so absolute tol won't do.
    DSI_EXPECT(recon.rows() == X.rows() && recon.cols() == X.cols());
    const double max_rel = ((recon - X).array().abs() / X.array().abs())
                                .maxCoeff();
    if (!(max_rel <= 1e-8)) {
        std::cerr << "FAIL: max relative error " << max_rel
                  << " > 1e-8\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

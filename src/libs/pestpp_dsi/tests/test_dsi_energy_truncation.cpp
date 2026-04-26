// test_dsi_energy_truncation.cpp
//
// Plan §10.1: with energy_threshold=t, n_components must equal the
// smallest k such that cumsum(s^2)[k-1] / sum(s^2) >= t. The argmax+1
// semantic must be matched (not searchsorted, which differs on ties).

#include "DsiEmulator.h"
#include "test_helpers.h"

#include <Eigen/SVD>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace {

int expected_components(const Eigen::VectorXd& s, double thresh) {
    if (thresh >= 1.0) return static_cast<int>(s.size());
    const double total = s.array().square().sum();
    if (total <= 0.0) return 1;
    double running = 0.0;
    const double cutoff = thresh * total;
    for (int i = 0; i < s.size(); ++i) {
        running += s(i) * s(i);
        if (running >= cutoff) return i + 1;
    }
    return static_cast<int>(s.size());
}

}  // namespace

int main() {
    using pestpp_dsi::DsiEmulator;

    constexpr int n_real = 50;
    constexpr int n_obs = 100;

    std::mt19937 rng(123u);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd X(n_real, n_obs);
    for (int i = 0; i < n_real; ++i)
        for (int j = 0; j < n_obs; ++j) X(i, j) = dist(rng);

    // Compute reference singular values.
    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::MatrixXd dev = X.rowwise() - mean;
    const double scale = std::sqrt(static_cast<double>(X.rows() - 1));
    Eigen::MatrixXd Z = dev / scale;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        Z, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& s = svd.singularValues();

    std::vector<std::string> names;
    names.reserve(n_obs);
    for (int j = 0; j < n_obs; ++j) names.push_back("o" + std::to_string(j));

    for (double thresh : {0.50, 0.75, 0.90, 0.99, 1.00}) {
        DsiEmulator::Config cfg;
        cfg.obs_names = names;
        cfg.energy_threshold = thresh;
        DsiEmulator dsi(cfg);
        dsi.fit(X);
        const int expect = expected_components(s, thresh);
        if (dsi.n_components() != expect) {
            std::cerr << "FAIL: thresh=" << thresh << " expected n_components="
                      << expect << " got=" << dsi.n_components() << "\n";
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

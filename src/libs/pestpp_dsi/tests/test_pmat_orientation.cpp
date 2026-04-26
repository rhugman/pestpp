// test_pmat_orientation.cpp
//
// Regression test for the V vs V^T trap called out in plan §4.2 step 6.
//
// pyemu computes us = np.dot(v.T, np.diag(s)) where numpy's `v` is V^T,
// so us == V * diag(s). Eigen's JacobiSVD::matrixV() returns V directly,
// so DsiEmulator must compute pmat = V * diag(s). If pmat were
// accidentally V^T * diag(s), pmat * pmat.T would not equal Z^T * Z.
//
// We verify pmat * pmat^T == Z^T * Z (with energy_threshold=1.0) on a
// fixed deterministic 5x4 input.

#include "DsiEmulator.h"
#include "test_helpers.h"

#include <cmath>
#include <vector>

int main() {
    using pestpp_dsi::DsiEmulator;

    Eigen::MatrixXd X(5, 4);
    X <<  1.0,  2.0,   3.0,   4.0,
          2.0,  4.0,   6.0,   8.0,
          0.5,  1.0,   1.5,   2.0,
          3.0,  6.0,   9.0,  12.5,
          1.5,  3.1,   4.6,   6.2;

    // Build expected Z^T * Z (no truncation).
    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::MatrixXd dev = X.rowwise() - mean;
    const double scale = std::sqrt(static_cast<double>(X.rows() - 1));
    Eigen::MatrixXd Z = dev / scale;
    Eigen::MatrixXd ZtZ = Z.transpose() * Z;

    DsiEmulator::Config cfg;
    cfg.obs_names = {"a", "b", "c", "d"};
    cfg.energy_threshold = 1.0;
    DsiEmulator dsi(cfg);
    dsi.fit(X);

    DSI_EXPECT(dsi.is_fitted());
    // n_components <= min(N-1, n_obs); for this 5x4 matrix the rank
    // can be at most 4.
    DSI_EXPECT(dsi.n_components() >= 1 && dsi.n_components() <= 4);

    Eigen::MatrixXd pmat = dsi.pmat();          // (n_obs x ncomp)
    DSI_EXPECT(pmat.rows() == 4);
    DSI_EXPECT(pmat.cols() == dsi.n_components());

    // pmat * pmat^T ≈ Z^T * Z when no truncation; if rank < n_obs the
    // equality holds only on the rank-revealed subspace, i.e. ZtZ
    // numerically ≈ pmat * pmat^T.
    Eigen::MatrixXd reconstructed = pmat * pmat.transpose();
    DSI_EXPECT_MATRIX_CLOSE(reconstructed, ZtZ, 1e-10);

    return EXIT_SUCCESS;
}

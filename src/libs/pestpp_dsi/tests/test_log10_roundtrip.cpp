// test_log10_roundtrip.cpp
// Forward then inverse Log10 must be identity to ~1e-12.

#include "DsiTransforms.h"
#include "test_helpers.h"

int main() {
    using pestpp_dsi::Log10Transform;

    Eigen::MatrixXd X(5, 4);
    X <<  1.0,   10.0,    100.0,    1000.0,
          0.1,   1.5,     12.3,     987.0,
          0.01,  0.001,   1e-4,     1e-5,
          1e6,   1e7,     1e8,      1e9,
          2.718, 3.14159, 1.41421,  0.5772;
    const Eigen::MatrixXd X_orig = X;

    Log10Transform t;       // empty cols_idx_ -> all columns
    t.fit(X);               // no-op
    t.apply(X);
    DSI_EXPECT(!std::isnan(X.array().abs().sum()));
    t.inverse(X);
    // Relative tolerance: log10 then exp(*ln(10)) accumulates ~few
    // ulps; values in this fixture span up to 1e9 so an absolute tol
    // of 1e-10 cannot be met. 1e-12 relative is the meaningful bar.
    const double max_rel_full =
        ((X - X_orig).array().abs() / X_orig.array().abs()).maxCoeff();
    DSI_EXPECT_NEAR(max_rel_full, 0.0, 1e-12);

    // Subset: only columns 1 and 3.
    Eigen::MatrixXd Y = X_orig;
    Log10Transform tsub({1, 3});
    tsub.apply(Y);
    // Untouched columns must equal original.
    DSI_EXPECT_NEAR((Y.col(0) - X_orig.col(0)).norm(), 0.0, 0.0);
    DSI_EXPECT_NEAR((Y.col(2) - X_orig.col(2)).norm(), 0.0, 0.0);
    // Log-transformed columns: pow back.
    DSI_EXPECT_MATRIX_CLOSE(
        Y.col(1), X_orig.col(1).array().log10().matrix(), 1e-12);
    tsub.inverse(Y);
    const double max_rel_sub =
        ((Y - X_orig).array().abs() / X_orig.array().abs()).maxCoeff();
    DSI_EXPECT_NEAR(max_rel_sub, 0.0, 1e-12);

    return EXIT_SUCCESS;
}

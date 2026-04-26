// test_rowwise_scaler.cpp
// Plan §10.1: 5x5 toy matrix, group of 3 columns, fit_groups subset
// of 2. Round-trip identity. Plus truth-scaler broadcast variant.

#include "DsiRowWiseScaler.h"
#include "test_helpers.h"

#include <vector>

int main() {
    using pestpp_dsi::DsiRowWiseScaler;

    // --- Training scaler: per-row min/max from fit_cols subset ------
    Eigen::MatrixXd X(5, 5);
    X <<  1.0,  2.0,  3.0,   4.0,   5.0,
          0.5,  1.5,  2.5,   3.5,   4.5,
         10.0, 20.0, 30.0,  40.0,  50.0,
         -1.0,  0.0,  1.0,   2.0,   3.0,
          7.0,  7.5,  7.7,   8.0,   8.2;

    DsiRowWiseScaler::GroupCols groups = {{"g", {0, 1, 2}}};        // act on cols 0..2
    DsiRowWiseScaler::GroupCols fit_groups = {{"g", {0, 1}}};       // min/max from cols 0..1 only

    DsiRowWiseScaler scaler({-1.0, 1.0}, groups, fit_groups);
    scaler.fit(X);

    Eigen::MatrixXd Y = X;
    scaler.apply(Y);

    // Row 0: fit_cols min=1, max=2; range=1. f_span=2.
    //   col 0: (1-1)/1*2 + (-1) = -1
    //   col 1: (2-1)/1*2 + (-1) = +1
    //   col 2: (3-1)/1*2 + (-1) = +3
    DSI_EXPECT_NEAR(Y(0, 0), -1.0, 1e-12);
    DSI_EXPECT_NEAR(Y(0, 1),  1.0, 1e-12);
    DSI_EXPECT_NEAR(Y(0, 2),  3.0, 1e-12);
    // Cols 3,4 untouched.
    DSI_EXPECT_NEAR(Y(0, 3), X(0, 3), 0.0);
    DSI_EXPECT_NEAR(Y(0, 4), X(0, 4), 0.0);

    // Round-trip.
    scaler.inverse(Y);
    DSI_EXPECT_MATRIX_CLOSE(Y, X, 1e-12);

    // --- Truth scaler: fit on a single-row, broadcast to many rows --
    Eigen::MatrixXd truth(1, 5);
    truth << 10.0, 20.0, 30.0, 40.0, 50.0;
    DsiRowWiseScaler tscaler({-1.0, 1.0},
                             {{"g", {0, 1, 2, 3}}},
                             {{"g", {0, 3}}});       // fit min=10, max=40
    tscaler.fit(truth);
    DSI_EXPECT(tscaler.n_fit_rows() == 1);

    // Apply on a 3-row matrix: per-group (10, 40) broadcasts to all rows.
    Eigen::MatrixXd many(3, 5);
    many << 10.0, 25.0, 30.0, 40.0,  100.0,
            10.0, 20.0, 30.0, 40.0,  -7.0,
            40.0, 25.0, 25.0, 10.0,   0.0;
    Eigen::MatrixXd many_orig = many;
    tscaler.apply(many);
    // Row 0, col 0: (10-10)/(40-10) * 2 + (-1) = -1
    DSI_EXPECT_NEAR(many(0, 0), -1.0, 1e-12);
    // Row 0, col 3: (40-10)/30 * 2 + (-1) = +1
    DSI_EXPECT_NEAR(many(0, 3),  1.0, 1e-12);
    // Col 4 untouched (not in groups).
    DSI_EXPECT_NEAR(many(0, 4), 100.0, 0.0);
    DSI_EXPECT_NEAR(many(2, 4),   0.0, 0.0);

    tscaler.inverse(many);
    DSI_EXPECT_MATRIX_CLOSE(many, many_orig, 1e-12);

    // --- Constant-row protection: range = 0 should not blow up ------
    Eigen::MatrixXd flat(2, 2);
    flat << 5.0, 5.0,
            7.0, 7.0;
    DsiRowWiseScaler::GroupCols all = {{"g", {0, 1}}};
    DsiRowWiseScaler s2({-1.0, 1.0}, all, all);
    s2.fit(flat);
    Eigen::MatrixXd flat2 = flat;
    s2.apply(flat2);
    // (5-5)/1 * 2 + (-1) = -1 for each entry.
    DSI_EXPECT_NEAR(flat2(0, 0), -1.0, 1e-12);
    DSI_EXPECT_NEAR(flat2(1, 1), -1.0, 1e-12);
    s2.inverse(flat2);
    // Inverse of (-1) under range=1 (constant-row) returns the row min,
    // i.e., the original constant value.
    DSI_EXPECT_NEAR(flat2(0, 0), 5.0, 1e-12);
    DSI_EXPECT_NEAR(flat2(1, 1), 7.0, 1e-12);

    return EXIT_SUCCESS;
}

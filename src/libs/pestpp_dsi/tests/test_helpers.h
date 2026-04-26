// test_helpers.h
//
// Minimal assertion helpers for pestpp_dsi unit tests. We deliberately
// avoid pulling in Catch2/gtest since pestpp itself has no test
// framework dependency and Phase 1 keeps the upstream-merge cost low.
//
// Each test is a standalone executable that returns 0 on success or
// non-zero on failure; CMake registers it via add_test().

#ifndef PESTPP_DSI_TEST_HELPERS_H
#define PESTPP_DSI_TEST_HELPERS_H

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#define DSI_EXPECT(cond)                                                    \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::cerr << "FAIL: " #cond " at " << __FILE__ << ":"          \
                      << __LINE__ << "\n";                                 \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

#define DSI_EXPECT_NEAR(actual, expected, tol)                              \
    do {                                                                     \
        const double _a = static_cast<double>(actual);                      \
        const double _e = static_cast<double>(expected);                    \
        const double _t = static_cast<double>(tol);                         \
        if (!(std::fabs(_a - _e) <= _t)) {                                  \
            std::cerr << "FAIL: |" #actual " - " #expected "| = "          \
                      << std::fabs(_a - _e) << " > " << _t << " at "       \
                      << __FILE__ << ":" << __LINE__                        \
                      << " (actual=" << _a << ", expected=" << _e << ")\n"; \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

inline bool dsi_matrix_close(const Eigen::MatrixXd& a,
                             const Eigen::MatrixXd& b,
                             double tol) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    return ((a - b).array().abs().maxCoeff() <= tol);
}

#define DSI_EXPECT_MATRIX_CLOSE(a, b, tol)                                  \
    do {                                                                     \
        const Eigen::MatrixXd& _A = (a);                                    \
        const Eigen::MatrixXd& _B = (b);                                    \
        if (_A.rows() != _B.rows() || _A.cols() != _B.cols()) {             \
            std::cerr << "FAIL: shape mismatch " #a " (" << _A.rows()       \
                      << "x" << _A.cols() << ") vs " #b " (" << _B.rows()  \
                      << "x" << _B.cols() << ") at " << __FILE__            \
                      << ":" << __LINE__ << "\n";                          \
            return EXIT_FAILURE;                                             \
        }                                                                    \
        const double _md = (_A - _B).array().abs().maxCoeff();              \
        if (!(_md <= (tol))) {                                              \
            std::cerr << "FAIL: max|" #a " - " #b "| = " << _md             \
                      << " > " << (tol) << " at " << __FILE__ << ":"        \
                      << __LINE__ << "\n";                                  \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

#endif  // PESTPP_DSI_TEST_HELPERS_H

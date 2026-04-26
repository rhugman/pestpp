// test_pipeline.cpp
// Pipeline: identity + log10 fit_apply / apply / inverse round-trip.

#include "DsiTransformPipeline.h"
#include "DsiTransforms.h"
#include "test_helpers.h"

int main() {
    using namespace pestpp_dsi;

    Eigen::MatrixXd X(4, 3);
    X << 1.0, 10.0, 0.5,
         2.0, 20.0, 1.5,
         3.0, 30.0, 2.5,
         4.0, 40.0, 3.5;
    const Eigen::MatrixXd X_orig = X;

    DsiTransformPipeline pipe;
    pipe.push_back(std::unique_ptr<DsiTransform>(new IdentityTransform()));
    pipe.push_back(std::unique_ptr<DsiTransform>(new Log10Transform()));
    DSI_EXPECT(!pipe.is_fitted());

    pipe.fit_apply(X);
    DSI_EXPECT(pipe.is_fitted());

    // After fit_apply, X should be log10(X_orig).
    DSI_EXPECT_MATRIX_CLOSE(X, X_orig.array().log10().matrix(), 1e-12);

    // Inverse must restore the original.
    pipe.inverse(X);
    DSI_EXPECT_MATRIX_CLOSE(X, X_orig, 1e-10);

    // Re-applying forward via apply() must give the same intermediate.
    Eigen::MatrixXd Y = X_orig;
    pipe.apply(Y);
    DSI_EXPECT_MATRIX_CLOSE(Y, X_orig.array().log10().matrix(), 1e-12);

    // Empty pipeline trivially identity.
    DsiTransformPipeline empty;
    Eigen::MatrixXd Z = X_orig;
    empty.fit_apply(Z);
    DSI_EXPECT_MATRIX_CLOSE(Z, X_orig, 0.0);
    empty.inverse(Z);
    DSI_EXPECT_MATRIX_CLOSE(Z, X_orig, 0.0);

    return EXIT_SUCCESS;
}

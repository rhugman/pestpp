// DsiTransformPipeline.h
//
// Ordered chain of DsiTransform objects. Forward path applies in push
// order; inverse path applies in reverse order. Mirrors the pyemu
// pipeline used by pyemu.emulators.DSI's transform list.
//
// Each transform is fit on the matrix that has *already* been
// transformed by the prior stages, exactly as pyemu does.

#ifndef PESTPP_DSI_TRANSFORM_PIPELINE_H
#define PESTPP_DSI_TRANSFORM_PIPELINE_H

#include "DsiTransforms.h"

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace pestpp_dsi {

class DsiTransformPipeline {
public:
    DsiTransformPipeline();

    void push_back(std::unique_ptr<DsiTransform> t);

    // Fit every transform in order on a copy of X, applying each
    // in-place so subsequent transforms see the prior output. After
    // this call, the pipeline is fitted.
    void fit_apply(Eigen::MatrixXd& X);

    // Forward transform (in place). Stages run in push order.
    void apply(Eigen::MatrixXd& X) const;

    // Inverse transform (in place). Stages run in reverse order.
    void inverse(Eigen::MatrixXd& X) const;

    bool empty() const { return transforms_.empty(); }
    std::size_t size() const { return transforms_.size(); }
    bool is_fitted() const { return fitted_; }

    // Returns the first staged NormalScoreTransform, or nullptr if
    // none was registered. Used by the golden-test loader and any
    // caller that needs to pre-seed NS state from disk; not used in
    // production paths.
    class NormalScoreTransform* find_normal_score_stage();

    // Mark the pipeline as fitted without invoking fit_apply. Used
    // when callers seed transform state externally (e.g. golden tests
    // pre-seeding NS state from a Python oracle).
    void mark_fitted_externally() { fitted_ = true; }

private:
    std::vector<std::unique_ptr<DsiTransform>> transforms_;
    bool fitted_ = false;
};

}  // namespace pestpp_dsi

#endif  // PESTPP_DSI_TRANSFORM_PIPELINE_H

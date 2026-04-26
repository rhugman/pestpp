#include "DsiTransformPipeline.h"

#include <stdexcept>

namespace pestpp_dsi {

DsiTransformPipeline::DsiTransformPipeline() = default;

void DsiTransformPipeline::push_back(std::unique_ptr<DsiTransform> t) {
    if (fitted_) {
        throw std::logic_error(
            "DsiTransformPipeline: cannot push transform after fit");
    }
    transforms_.push_back(std::move(t));
}

void DsiTransformPipeline::fit_apply(Eigen::MatrixXd& X) {
    for (auto& t : transforms_) {
        t->fit(X);
        t->apply(X);
    }
    fitted_ = true;
}

void DsiTransformPipeline::apply(Eigen::MatrixXd& X) const {
    if (!fitted_) {
        throw std::logic_error(
            "DsiTransformPipeline::apply called before fit_apply");
    }
    for (auto& t : transforms_) {
        t->apply(X);
    }
}

void DsiTransformPipeline::inverse(Eigen::MatrixXd& X) const {
    if (!fitted_) {
        throw std::logic_error(
            "DsiTransformPipeline::inverse called before fit_apply");
    }
    for (auto it = transforms_.rbegin(); it != transforms_.rend(); ++it) {
        (*it)->inverse(X);
    }
}

NormalScoreTransform* DsiTransformPipeline::find_normal_score_stage() {
    for (auto& t : transforms_) {
        if (auto* p = dynamic_cast<NormalScoreTransform*>(t.get())) {
            return p;
        }
    }
    return nullptr;
}

}  // namespace pestpp_dsi

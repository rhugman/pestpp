// test_active_refine_stop.cpp
// Documents the stopping predicate of the active-refine probe loop in
// EnsembleMethod::active_refine_lambda_candidates (see
// EnsembleMethodUtils.cpp, search for "active_refine probe loop").
//
// The loop stops when:
//     min_pred_remaining >= best_actual_phi_so_far
// i.e., the DSI-predicted phi for the best UNPROBED candidate is no
// better than the best phi we have already FOM-validated. At that
// point continuing to probe cannot improve the winner choice.
//
// This file is a sanity test of the predicate alone; the full probe
// loop (FOM eval, training-store append, refit) is exercised by the
// PAIR-07 integration validation, not here.

#include "test_helpers.h"
#include <limits>
#include <vector>

namespace {

// Returns true if the active-refine loop should STOP.
bool should_stop(double best_actual_so_far,
                 const std::vector<double>& predicted_phi_remaining)
{
    if (predicted_phi_remaining.empty()) return true;  // nothing left
    double min_pred = std::numeric_limits<double>::infinity();
    for (double v : predicted_phi_remaining)
        if (std::isfinite(v) && v < min_pred) min_pred = v;
    if (!std::isfinite(min_pred)) return true;
    if (!std::isfinite(best_actual_so_far)) return false;  // no probe yet
    return min_pred >= best_actual_so_far;
}

} // namespace

int main() {
    // No probe yet -> don't stop, regardless of remaining predictions.
    DSI_EXPECT(!should_stop(std::numeric_limits<double>::infinity(),
                            {1.0, 5.0, 50.0}));

    // Best probed = 10. Remaining min predicted = 1 -> keep going.
    DSI_EXPECT(!should_stop(10.0, {1.0, 5.0, 100.0}));

    // Best probed = 10. Remaining min predicted = 11 -> stop.
    DSI_EXPECT(should_stop(10.0, {11.0, 50.0, 200.0}));

    // Best probed = 10. Remaining min = 10 exactly -> stop (tie; >=).
    DSI_EXPECT(should_stop(10.0, {10.0, 50.0}));

    // Best probed = 10. Remaining min = 9.999 -> keep going.
    DSI_EXPECT(!should_stop(10.0, {9.999, 50.0}));

    // Empty remaining list -> stop (no candidates left to probe).
    DSI_EXPECT(should_stop(10.0, {}));

    // All non-finite predictions -> stop (no information).
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const double Inf = std::numeric_limits<double>::infinity();
    DSI_EXPECT(should_stop(10.0, {NaN, Inf, NaN}));

    // Mixed: only finite remaining is below best -> keep going.
    DSI_EXPECT(!should_stop(10.0, {NaN, 5.0, Inf}));

    // Truth_07-style: best probed high (info-poor prior), DSI for
    // remaining higher still -> stop.
    DSI_EXPECT(should_stop(2.0e9, {2.5e9, 3.0e9, 4.0e9}));

    return 0;
}

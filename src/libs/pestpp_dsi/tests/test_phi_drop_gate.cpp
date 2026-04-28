// test_phi_drop_gate.cpp
// Sanity-check the predicate of the prior-phi sanity gate.
//
// The gate (see EnsembleMethodUtils.cpp, search for
// "Prior-phi sanity gate") evaluates, after the DSI surrogate has
// produced predicted phi for each lambda candidate:
//
//     predicate = (max_phi_drop_factor > 0)
//              && (last_best_mean > 0)
//              && (min(predicted_phi) < last_best_mean / max_phi_drop_factor)
//
// When the predicate is true the surrogate ranking is discarded and
// the iter falls back to FOM lambda testing. This file documents the
// expected predicate output for representative scenarios so a future
// reader can spot a sign-flip / off-by-one regression.

#include "test_helpers.h"
#include <limits>
#include <vector>

namespace {

bool gate_triggers(double last_best_mean,
                   double max_phi_drop_factor,
                   const std::vector<double>& predicted_phi)
{
    if (max_phi_drop_factor <= 0.0) return false;
    if (last_best_mean <= 0.0) return false;
    if (predicted_phi.empty()) return false;
    double min_pred = std::numeric_limits<double>::infinity();
    for (double v : predicted_phi)
        if (std::isfinite(v) && v < min_pred) min_pred = v;
    if (!std::isfinite(min_pred)) return false;
    return min_pred < last_best_mean / max_phi_drop_factor;
}

} // namespace

int main() {
    // Disabled (default) — never triggers.
    DSI_EXPECT(!gate_triggers(100.0, 0.0, {1.0, 5.0, 50.0}));

    // Negative factor — never triggers.
    DSI_EXPECT(!gate_triggers(100.0, -1.0, {1.0}));

    // Zero current phi (degenerate prior; nothing to compare against).
    DSI_EXPECT(!gate_triggers(0.0, 10.0, {1.0}));

    // Empty predicted vector — defensive; never triggers.
    DSI_EXPECT(!gate_triggers(100.0, 10.0, {}));

    // Predicted phi reaches the threshold exactly — strict less-than,
    // so does NOT trigger (consistent with the C++ "<").
    DSI_EXPECT(!gate_triggers(100.0, 10.0, {10.0, 20.0, 30.0}));

    // Headline trigger from the truth_07 diagnostic: current_phi 1e2,
    // factor 10 → threshold 10. Predicted minimum 1.0 (claims 100×
    // drop, well over the allowed 10× per-iter ceiling).
    DSI_EXPECT(gate_triggers(100.0, 10.0, {1.0, 5.0, 50.0}));

    // Conservative factor 2 — a 10× predicted drop now also trips.
    DSI_EXPECT(gate_triggers(100.0, 2.0, {30.0, 40.0}));

    // NaN / inf entries are ignored when computing the minimum.
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const double Inf = std::numeric_limits<double>::infinity();
    DSI_EXPECT(gate_triggers(100.0, 10.0, {NaN, Inf, 5.0}));
    DSI_EXPECT(!gate_triggers(100.0, 10.0, {NaN, Inf, 50.0}));

    // All entries non-finite → no minimum → no trigger.
    DSI_EXPECT(!gate_triggers(100.0, 10.0, {NaN, Inf}));

    // Predicted minimum just below threshold — still trips.
    DSI_EXPECT(gate_triggers(100.0, 10.0, {9.999, 11.0}));

    return 0;
}

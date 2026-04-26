// test_training_store.cpp
// Append two batches; check matrix shape, ordering, and column-reorder lookup.

#include "DsiTrainingStore.h"
#include "test_helpers.h"

#include <vector>

int main() {
    using pestpp_dsi::DsiTrainingStore;

    DsiTrainingStore store;
    DSI_EXPECT(store.n_realisations() == 0);
    DSI_EXPECT(store.n_obs() == 0);

    std::vector<std::string> names = {"oa", "ob", "oc"};
    Eigen::MatrixXd batch1(2, 3);
    batch1 << 1.0, 2.0, 3.0,
              4.0, 5.0, 6.0;
    store.append(batch1, names);
    DSI_EXPECT(store.n_realisations() == 2);
    DSI_EXPECT(store.n_obs() == 3);
    DSI_EXPECT(store.obs_names() == names);

    Eigen::MatrixXd batch2(3, 3);
    batch2 << 7.0,  8.0,  9.0,
              10.0, 11.0, 12.0,
              13.0, 14.0, 15.0;
    store.append(batch2, names);
    DSI_EXPECT(store.n_realisations() == 5);

    Eigen::MatrixXd combined(5, 3);
    combined << batch1, batch2;
    DSI_EXPECT_MATRIX_CLOSE(store.matrix(), combined, 0.0);

    // matrix_for in the same order is a noop.
    DSI_EXPECT_MATRIX_CLOSE(store.matrix_for(names), combined, 0.0);

    // matrix_for in a different order rearranges columns correctly.
    std::vector<std::string> reordered = {"oc", "oa", "ob"};
    Eigen::MatrixXd expected(5, 3);
    expected.col(0) = combined.col(2);
    expected.col(1) = combined.col(0);
    expected.col(2) = combined.col(1);
    DSI_EXPECT_MATRIX_CLOSE(store.matrix_for(reordered), expected, 0.0);

    // Mismatched names must throw.
    bool threw = false;
    try {
        store.matrix_for({"oa", "od", "oc"});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    DSI_EXPECT(threw);

    // Append with different names must throw.
    threw = false;
    try {
        store.append(batch1, {"oa", "ob", "od"});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    DSI_EXPECT(threw);

    return EXIT_SUCCESS;
}

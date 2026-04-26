// test_golden_pyemu.cpp
//
// Phase 2 exit criterion (plan §10.2 + §12 Phase 2 exit).
//
// Loads artefacts produced by tests/golden/run_python_oracle.py and
// compares the C++ DsiEmulator output against pyemu's reference. The
// NS state per column is pre-seeded from the Python oracle, sidestepping
// the std::normal_distribution cross-compiler portability issue
// (plan §5.2). Asserts:
//   - pmat (n_obs x ncomp) max abs diff <= 1e-10 after sign alignment
//   - s_kept max abs diff <= 1e-10
//   - ovals max abs diff <= 1e-12
//   - predict(latent) max abs diff <= 1e-8 (plan §10.4 spec)
//
// Usage: test_golden_pyemu <artefact_dir>
// Returns 77 (CTest skip code) when the directory or any artefact
// is missing — lets ctest report SKIP rather than FAIL when the
// oracle has not been pre-run.

#include "DsiEmulator.h"
#include "DsiTransforms.h"
#include "test_helpers.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int SKIP_CODE = 77;

bool path_exists(const std::string& p) {
    std::ifstream f(p);
    return f.good();
}

Eigen::MatrixXd read_matrix(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> row;
        double x;
        while (iss >> x) row.push_back(x);
        if (!row.empty()) rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::runtime_error("empty matrix " + path);
    const int n_cols = static_cast<int>(rows[0].size());
    Eigen::MatrixXd M(rows.size(), n_cols);
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        if (static_cast<int>(rows[i].size()) != n_cols) {
            throw std::runtime_error(
                "ragged row in " + path + " at line " + std::to_string(i));
        }
        for (int j = 0; j < n_cols; ++j) M(i, j) = rows[i][j];
    }
    return M;
}

Eigen::VectorXd read_vector(const std::string& path) {
    Eigen::MatrixXd M = read_matrix(path);
    if (M.cols() != 1) {
        throw std::runtime_error("expected 1 col in " + path);
    }
    return M.col(0);
}

struct OracleMeta {
    int n_train = 0;
    int n_obs = 0;
    int n_components = 0;
    int n_test = 0;
    std::vector<std::string> obs_names;
};

OracleMeta read_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    OracleMeta m;
    f >> m.n_train >> m.n_obs >> m.n_components >> m.n_test;
    if (!f) throw std::runtime_error("malformed meta header in " + path);
    std::string rest;
    std::getline(f, rest);
    std::string nm;
    while (std::getline(f, nm)) {
        if (!nm.empty()) m.obs_names.push_back(nm);
    }
    if (static_cast<int>(m.obs_names.size()) != m.n_obs) {
        throw std::runtime_error("meta obs_names count mismatch with n_obs");
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <artefact_dir>\n";
        return EXIT_FAILURE;
    }
    const std::string dir = argv[1];

    if (!path_exists(dir + "/meta.txt")) {
        std::cerr << "SKIP: oracle artefacts missing at " << dir
                  << " (run tests/golden/run_python_oracle.py first)\n";
        return SKIP_CODE;
    }

    using pestpp_dsi::DsiEmulator;
    using pestpp_dsi::NormalScoreTransform;

    OracleMeta meta;
    Eigen::MatrixXd train, latent, predicted_py, predicted_hp_py, pmat_py;
    Eigen::VectorXd ovals_py, s_kept_py;
    try {
        meta = read_meta(dir + "/meta.txt");
        train = read_matrix(dir + "/train.txt");
        latent = read_matrix(dir + "/latent.txt");
        predicted_py = read_matrix(dir + "/predicted.txt");
        predicted_hp_py = read_matrix(dir + "/predicted_hp.txt");
        ovals_py = read_vector(dir + "/ovals.txt");
        s_kept_py = read_vector(dir + "/s_kept.txt");
        pmat_py = read_matrix(dir + "/pmat.txt");
    } catch (const std::exception& ex) {
        std::cerr << "load error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    DSI_EXPECT(train.rows() == meta.n_train);
    DSI_EXPECT(train.cols() == meta.n_obs);
    DSI_EXPECT(latent.rows() == meta.n_test);
    DSI_EXPECT(latent.cols() == meta.n_components);
    DSI_EXPECT(predicted_py.rows() == meta.n_test);
    DSI_EXPECT(predicted_py.cols() == meta.n_obs);
    DSI_EXPECT(ovals_py.size() == meta.n_obs);
    DSI_EXPECT(s_kept_py.size() == meta.n_components);
    DSI_EXPECT(pmat_py.rows() == meta.n_obs);
    DSI_EXPECT(pmat_py.cols() == meta.n_components);

    // Build emulator with a single NormalScore stage spanning all cols.
    DsiEmulator::Config cfg;
    cfg.obs_names = meta.obs_names;
    cfg.energy_threshold = 1.0;
    DsiEmulator::TransformSpec spec;
    spec.kind = DsiEmulator::TransformSpec::Kind::NormalScore;
    cfg.transforms.push_back(spec);
    DsiEmulator dsi(cfg);

    NormalScoreTransform* ns = dsi.normal_score_stage();
    DSI_EXPECT(ns != nullptr);
    for (int j = 0; j < meta.n_obs; ++j) {
        const std::string ns_path = dir + "/ns_" + meta.obs_names[j] + ".txt";
        Eigen::MatrixXd zo;
        try {
            zo = read_matrix(ns_path);
        } catch (const std::exception& ex) {
            std::cerr << ex.what() << "\n";
            return SKIP_CODE;
        }
        DSI_EXPECT(zo.cols() == 2);
        ns->set_state(j, zo.col(0).eval(), zo.col(1).eval());
    }
    // Mark the pipeline fitted (transform state is now populated).
    dsi.mark_pipeline_fitted_externally();

    // Run the SVD half on the (raw) training data with NS forward
    // applied via the now-seeded pipeline.
    dsi.fit_from_transformed(train, /*also_apply_pipeline_forward=*/true);

    // --- pmat agreement ----------------------------------------------
    DSI_EXPECT(dsi.n_components() == meta.n_components);
    const Eigen::MatrixXd& pmat_cpp = dsi.pmat();
    DSI_EXPECT(pmat_cpp.rows() == pmat_py.rows());
    DSI_EXPECT(pmat_cpp.cols() == pmat_py.cols());

    // SVD has a per-column sign ambiguity. Resolve by aligning each
    // column's sign to maximise positive correlation against pyemu's
    // pmat. The signs cancel through `predict()` (pmat * latent),
    // so we apply the same flip to `latent` for the predict check.
    Eigen::VectorXd col_sign(pmat_cpp.cols());
    for (Eigen::Index k = 0; k < pmat_cpp.cols(); ++k) {
        col_sign(k) = pmat_cpp.col(k).dot(pmat_py.col(k)) >= 0 ? 1.0 : -1.0;
    }
    Eigen::MatrixXd pmat_aligned = pmat_cpp;
    for (Eigen::Index k = 0; k < pmat_aligned.cols(); ++k) {
        pmat_aligned.col(k) *= col_sign(k);
    }
    const double pmat_max_abs = (pmat_aligned - pmat_py).array().abs().maxCoeff();
    if (pmat_max_abs > 1e-10) {
        std::cerr << "FAIL: pmat max abs diff = " << pmat_max_abs
                  << " > 1e-10\n";
        return EXIT_FAILURE;
    }
    const double s_max_abs = (dsi.singular_values() - s_kept_py)
                                  .array().abs().maxCoeff();
    if (s_max_abs > 1e-10) {
        std::cerr << "FAIL: s_kept max abs diff = " << s_max_abs
                  << " > 1e-10\n";
        return EXIT_FAILURE;
    }
    const double ovals_max_abs = (dsi.ovals() - ovals_py).array().abs().maxCoeff();
    if (ovals_max_abs > 1e-12) {
        std::cerr << "FAIL: ovals max abs diff = " << ovals_max_abs
                  << " > 1e-12\n";
        return EXIT_FAILURE;
    }

    // --- predict() agreement -----------------------------------------
    Eigen::MatrixXd latent_aligned = latent;
    for (Eigen::Index k = 0; k < latent_aligned.cols(); ++k) {
        latent_aligned.col(k) *= col_sign(k);
    }
    // --- Intermediate sim_t agreement (NS-z space) -------------------
    Eigen::MatrixXd sim_t_py = read_matrix(dir + "/sim_t.txt");  // (n_test, n_obs)
    Eigen::MatrixXd sim_t_cpp = (dsi.pmat() * latent_aligned.transpose()).colwise()
                                + dsi.ovals();                     // (n_obs, n_test)
    sim_t_cpp.transposeInPlace();                                   // (n_test, n_obs)
    const double sim_t_max_abs = (sim_t_cpp - sim_t_py).array().abs().maxCoeff();
    if (sim_t_max_abs > 1e-9) {
        std::cerr << "FAIL: sim_t (NS-z space) max abs diff = "
                  << sim_t_max_abs << " > 1e-9\n";
        return EXIT_FAILURE;
    }

    Eigen::MatrixXd predicted_cpp = dsi.predict(latent_aligned);
    DSI_EXPECT(predicted_cpp.rows() == predicted_py.rows());
    DSI_EXPECT(predicted_cpp.cols() == predicted_py.cols());

    // Compare against pyemu's *high-precision* predict (NS inverse run
    // in float64 — bypasses the float32 downcast in
    // TransformerPipeline.inverse_transform at transformers.py:743 +
    // 753). The standard predict reference is also reported for
    // diagnostic context but only constrained to a much looser bar
    // (~1e-6) since pyemu's float32 downcast caps relative precision
    // around 6e-8 by construction.
    const double pred_hp_max_abs =
        (predicted_cpp - predicted_hp_py).array().abs().maxCoeff();
    const double pred_lp_max_abs =
        (predicted_cpp - predicted_py).array().abs().maxCoeff();

    if (pred_hp_max_abs > 1e-10) {
        Eigen::Index wr = 0, wc = 0;
        (predicted_cpp - predicted_hp_py).array().abs().maxCoeff(&wr, &wc);
        std::cerr << "FAIL: predict (vs HP) max abs diff = "
                  << pred_hp_max_abs << " > 1e-10"
                  << " at row=" << wr << " col=" << wc
                  << " (cpp=" << predicted_cpp(wr, wc)
                  << ", hp=" << predicted_hp_py(wr, wc) << ")\n";
        return EXIT_FAILURE;
    }
    if (pred_lp_max_abs > 1e-6) {
        std::cerr << "FAIL: predict (vs LP/float32) max abs diff = "
                  << pred_lp_max_abs << " > 1e-6 — exceeds the float32 "
                  << "downcast floor in pyemu's TransformerPipeline.\n";
        return EXIT_FAILURE;
    }

    std::cerr << "OK  golden vs pyemu: pmat_max_abs=" << pmat_max_abs
              << " s_max_abs=" << s_max_abs
              << " ovals_max_abs=" << ovals_max_abs
              << " sim_t_max_abs=" << sim_t_max_abs
              << " pred_hp_max_abs=" << pred_hp_max_abs
              << " pred_lp_max_abs=" << pred_lp_max_abs
              << "  (LP path bounded by pyemu float32 downcast)\n";
    return EXIT_SUCCESS;
}

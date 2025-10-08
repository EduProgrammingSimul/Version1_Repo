# PWR Controller Optimization & Monitoring Workbench (DTAF v3.0)

## Description

This workbench delivers an end-to-end toolkit for designing, optimizing, and validating
pressurized water reactor (PWR) turbine governor controllers against realistic grid
conditions. It unifies high-fidelity simulation models, configurable optimization
pipelines, and streamlined reporting so researchers and engineers can iterate on control
strategies with confidence.

## 1. Project Overview

This project provides a comprehensive Python-based simulation framework for modeling the interaction between a Pressurized Water Reactor (PWR) nuclear power plant and the electrical grid. It focuses on the development, robust optimization, automated validation, and comparative analysis of advanced steam turbine governor control strategies.

The primary goal of this framework is to engineer and evaluate controllers (PID, FLC, RL) that are not just "optimal" under ideal conditions, but are robustly stable and reliable across a wide range of challenging operational and off-normal scenarios.

### Key Features

* **High-Fidelity Simulation Environment:** A modular `PWRGymEnvUnified` environment, compatible with the Gymnasium standard, that accurately models core physics, turbine dynamics, and grid interaction.
* **Advanced Controller Suite:** Implementations for PID, Fuzzy Logic (FLC), and a sophisticated Reinforcement Learning (RL) agent.
* **Robust Optimization Suite:** State-of-the-art optimizers for PID and FLC controllers that tune parameters against a full suite of validation scenarios to guarantee robustness.
* **Advanced RL Training:** A dedicated training pipeline for the RL agent, featuring a multi-stage curriculum to progressively increase task difficulty and ensure stable learning.
* **Automated End-to-End Validation & Reporting:** Automatic generation of comprehensive Markdown reports with advanced metrics and plots after any optimization or training run.
* **Interactive Streamlit UI:** A professional user interface for initiating controller optimization, training, analysis, and live simulation monitoring.

## 2. Project Architecture

The framework is designed with a clear separation of concerns, ensuring maintainability and robustness across its core components: `models`, `environment`, `controllers`, `optimization_suite`, `analysis`.

## 3. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <project-directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 4. How to use workbench

All critical parameters are centralized in a single "source of truth" file for reliability and ease of modification:

* **`config/parameters.py`**: This file contains all core physics constants, default controller settings, safety limits, and the hyperparameters for the RL agent training process.
### Multi-Seed Evaluation & Reporting

The evaluation pipeline now supports running deterministic rollouts across multiple seeds and aggregating statistical summaries:

```bash
$env:PYTHONPATH='.';
python scripts/full_eval.py --out results --suite-file config/scenario_suite.txt --controllers PID FLC RL --eval-seed 42 --eval-seeds 101 202 303
python scripts/validate_and_report.py --controllers PID FLC RL --scenarios all --out results --eval-seed 42 --eval-seeds 101 202 303 --bootstrap-samples 2000
```

* Individual runs are written to `results/seed_<seed>/validation/` to preserve provenance.
* Aggregated artefacts (metrics means, standard deviation, paired statistical tests, frequency-domain summaries, and bootstrap confidence intervals) are emitted under `results/metrics/`.
* Radar and time-series visualisations can consume the generated `perf_profile_7d_ci.csv` and other summary files to render uncertainty bands.

### Frequency-Probe Scenarios

Three excitation-oriented scenarios (`prbs_excitation_probe`, `multisine_frequency_probe`, and `actuator_saturation_challenge`) have been added to exercise closed-loop bandwidth, coherence, and actuator margin metrics.


### New Analysis Artefacts

The validation report now includes:

* `paired_tests.csv` – Wilcoxon signed-rank tests with FDR-adjusted p-values and effect sizes.
* `frequency_domain_metrics.csv` – PSD-derived indicators and damping estimates for excitation scenarios.
* `perf_profile_7d_ci.csv` – bootstrap confidence intervals for the seven composite dimensions used in radar plots.
* `combined_metrics_summary.csv` – per-metric mean, standard deviation, and sample counts across seeds.

## 5. Contributing Guidelines

We welcome contributions that improve controller strategies, analysis tooling, documentation, and UI/UX. To streamline reviews and ensure consistent quality across the project, please follow these steps when opening a pull request:

1. **Open an Issue First:** Describe the bug, enhancement, or feature you plan to work on so maintainers can provide guidance or flag duplicates.
2. **Create a Dedicated Branch:** Use a descriptive branch name (for example, `feature/new-probe-scenario` or `fix/pid-saturation`).
3. **Add Tests & Documentation:** Update or create unit tests, validation scenarios, and README/Docs sections relevant to your change.
4. **Run the Validation Suite:** Execute the workflows under `scripts/` or `run_training.py` where applicable to ensure no regressions are introduced.
5. **Follow Coding Standards:** Adhere to the existing code style and linting configurations defined in `requirements.txt` and styled configs across the repository.
6. **Submit a Detailed PR:** Summarize your changes, include validation evidence (logs, plots, metrics), and request review from relevant subject-matter maintainers.

For any questions, feel free to start a discussion or reach out to the maintainers listed in the repository insights.

import logging
import streamlit as st
import sys
import os
import subprocess
import yaml
import time
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List
import glob
from datetime import datetime


try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from analysis.parameter_manager import ParameterManager
    from analysis.scenario_definitions import get_scenarios, constant_load
    from analysis.scenario_executor import ScenarioExecutor
    from main_analysis import load_controller
    from environment.pwr_gym_env import PWRGymEnvUnified

except ImportError as e:
    st.error(f"FATAL ERROR: Could not import necessary project modules: {e}")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


st.set_page_config(layout="wide", page_title="PWR Control Workbench v2.2")
st.title("🔬 PWR Controller Optimization & Monitoring Workbench")


def initialize_session_state():

    state_defaults = {
        "is_monitoring": False,
        "stop_monitoring_flag": False,
        "monitoring_status": "Idle",
        "monitoring_controller_instance": None,
        "selected_scenario_config": None,
        "live_params": {},
        "params_updated_signal": False,
        "last_monitored_ctrl": None,
        "manual_valve_command": 0.5,
        "chart_history": {
            "time_s": deque(maxlen=200),
            "grid_frequency_hz": deque(maxlen=200),
            "turbine_speed_rpm": deque(maxlen=200),
            "reactor_power_mw": deque(maxlen=200),
            "valve_actual": deque(maxlen=200),
        },
    }
    for key, value in state_defaults.items():
        st.session_state.setdefault(key, value)


initialize_session_state()


@st.cache_resource
def load_base_config_and_scenarios(root_path):

    logger.info("Loading base configuration and scenarios...")
    config_file_path = os.path.join(root_path, "config", "parameters.py")
    param_manager = ParameterManager(config_filepath=config_file_path)
    full_config = param_manager.get_all_parameters()
    core_config = full_config.get("CORE_PARAMETERS")
    if not core_config:
        raise ValueError("'CORE_PARAMETERS' key missing from configuration.")
    scenarios = get_scenarios(core_config)
    optimized_dir = os.path.join(root_path, "config", "optimized_controllers")
    results_dir = os.path.join(root_path, "results", "monitoring_plots")
    os.makedirs(results_dir, exist_ok=True)
    return full_config, core_config, scenarios, optimized_dir, results_dir


(
    full_config,
    core_config,
    scenarios_config_dict,
    OPTIMIZED_CONTROLLERS_DIR,
    RESULTS_DIR,
) = load_base_config_and_scenarios(project_root)


with st.sidebar:
    st.header("✅ Select Task")
    task = st.radio(
        "Choose Action:",
        ["Live Monitoring", "Optimize Controller", "Run Comparative Analysis"],
    )
    st.divider()

    st.header("🔧 Optimized Configurations")

    if os.path.exists(OPTIMIZED_CONTROLLERS_DIR):
        for filename in sorted(os.listdir(OPTIMIZED_CONTROLLERS_DIR)):
            if filename.endswith(".yaml"):
                with st.expander(f"⚙️ {filename}"):
                    try:
                        with open(
                            os.path.join(OPTIMIZED_CONTROLLERS_DIR, filename), "r"
                        ) as f:
                            st.json(yaml.safe_load(f))
                    except Exception as e:
                        st.error(f"Failed to read {filename}")
            elif filename.endswith(".zip"):
                st.text(f"🤖 {filename}")

    st.divider()
    st.header("📈 Validation Reports")
    report_files = sorted(
        glob.glob(os.path.join(project_root, "results", "reports", "*.md")),
        key=os.path.getmtime,
        reverse=True,
    )
    if report_files:
        for report_path in report_files[:5]:
            with st.expander(f"📄 {os.path.basename(report_path)}"):
                st.download_button(
                    "Download",
                    data=open(report_path, "rb").read(),
                    file_name=os.path.basename(report_path),
                )
    else:
        st.caption("No reports found.")


if task == "Live Monitoring":
    st.subheader("🚀 Live Simulation Monitoring & Tuning")

    col1, col2 = st.columns(2)
    with col1:
        controller_options = (
            ["No Controller"]
            + [os.path.splitext(f)[0] for f in os.listdir(OPTIMIZED_CONTROLLERS_DIR)]
            + ["PID", "FLC"]
        )
        selected_controller_name = st.selectbox(
            "Select Controller:", sorted(list(set(controller_options)))
        )

    with col2:
        scenario_options = ["No Scenario"] + list(scenarios_config_dict.keys())
        selected_scenario_name = st.selectbox("Select Scenario:", scenario_options)

    st.info(
        "Live Monitoring UI is selected. Controls for starting, stopping, and viewing the simulation would appear here."
    )


elif task == "Optimize Controller":
    st.subheader("🛠️ Controller Optimization / Training")

    controller_to_opt = st.selectbox(
        "Select Controller to Optimize:", ["PID", "FLC", "RL_Agent"]
    )
    log_placeholder = st.empty()

    if st.button(
        f"🚀 Start {controller_to_opt} Optimization",
        key=f"start_{controller_to_opt}_opt",
    ):
        with st.spinner(
            f"Running optimization for {controller_to_opt}... This may take a while."
        ):
            args = ["--controller", controller_to_opt]
            if controller_to_opt == "PID":
                args.append("--global_pid_de")

            process = subprocess.Popen(
                [sys.executable, os.path.join(project_root, "run_optimization.py")]
                + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            with log_placeholder.expander("Optimization Log", expanded=True):
                log_area = st.empty()
                log_text = ""
                for line in iter(process.stdout.readline, ""):
                    log_text += line
                    log_area.code(log_text)
                process.wait()

            if process.returncode == 0:
                st.success(f"{controller_to_opt} optimization finished successfully!")
                st.experimental_rerun()
            else:
                st.error(
                    f"{controller_to_opt} optimization failed. Check logs for details."
                )


elif task == "Run Comparative Analysis":
    st.subheader("📊 Run Comparative Analysis")

    all_controllers = ["PID", "FLC", "RL_Agent"] + [
        os.path.splitext(f)[0] for f in os.listdir(OPTIMIZED_CONTROLLERS_DIR)
    ]
    controllers_to_compare = st.multiselect(
        "Select controllers to compare:",
        sorted(list(set(all_controllers))),
        default=sorted(list(set(all_controllers))),
    )

    if st.button("🚀 Launch Full Analysis"):
        with st.spinner("Running full comparative analysis..."):
            log_placeholder_analysis = st.empty()
            args = ["--controllers"] + controllers_to_compare
            process = subprocess.Popen(
                [sys.executable, os.path.join(project_root, "main_analysis.py")] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            with log_placeholder_analysis.expander("Analysis Log", expanded=True):
                log_area = st.empty()
                log_text = ""
                for line in iter(process.stdout.readline, ""):
                    log_text += line
                    log_area.code(log_text)
                process.wait()

            if process.returncode == 0:
                st.success(
                    "Analysis finished. The latest report is now available in the sidebar."
                )
                st.experimental_rerun()
            else:
                st.error("Analysis failed. Check logs.")

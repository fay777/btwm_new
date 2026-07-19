from __future__ import annotations

import os
import pathlib
import py_compile
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
DREAMERV3_ROOT = ROOT / "dreamerv3"

if str(DREAMERV3_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMERV3_ROOT))


def check_syntax() -> None:
    print("[1/4] Checking syntax for key modules")
    for relpath in (
        "code/btwm_head.py",
        "dreamerv3/dreamerv3/agent.py",
        "dreamerv3/dreamerv3/rssm.py",
    ):
        py_compile.compile(str(ROOT / relpath), doraise=True)


def run_unit_tests_if_possible() -> None:
    print("[2/4] Running BTWM unit tests if dependencies are available")
    try:
        __import__("elements")
        __import__("embodied")
    except ModuleNotFoundError:
        print("Skipping unit tests: required local/runtime modules are not available in the current environment.")
        return
    env = os.environ.copy()
    pythonpath = [str(DREAMERV3_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    subprocess.run(
        [sys.executable, "-m", "unittest", "code.test_btwm_head"],
        cwd=ROOT,
        check=True,
        env=env,
    )


def build_experiment_command(model: str, domain: str, task: str) -> list[str]:
    task_name = task
    replay_size = "300000"
    if domain == "atari":
        config = "atari100k"
        task_name = f"atari100k_{task}"
        replay_size = "500000"
    elif domain == "dmc":
        config = "dmc_vision"
        task_name = f"dmc_{task}"
    elif domain == "crafter":
        config = "crafter"
        task_name = f"crafter_{task}"
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    cmd = [
        sys.executable,
        str(ROOT / "dreamerv3" / "dreamerv3" / "main.py"),
        "--logdir",
        str(ROOT / "runs" / "smoke_dryrun"),
        "--configs",
        config,
        "--task",
        task_name,
        "--seed",
        "0",
        "--run.steps",
        "1000000",
        "--replay.size",
        replay_size,
    ]
    if model == "btwm":
        cmd += [
            "--agent.use_btwm",
            "True",
            "--agent.inv_loss_weight",
            "0.05",
            "--agent.inv_head_loss_weight",
            "0.0",
            "--agent.inv_num_bins",
            "21",
            "--agent.inv_confidence_gating",
            "False",
            "--agent.inv_policy_weight",
            "0.0",
        ]
    elif model == "dreamerv3":
        cmd += [
            "--agent.use_btwm",
            "True",
            "--agent.inv_loss_weight",
            "0.0",
            "--agent.inv_head_loss_weight",
            "0.0",
            "--agent.inv_num_bins",
            "21",
            "--agent.inv_confidence_gating",
            "False",
            "--agent.inv_policy_weight",
            "0.0",
        ]
    else:
        raise ValueError(f"Unsupported model: {model}")
    return cmd


def verify_command_generation() -> None:
    print("[3/4] Verifying experiment command generation")
    for spec in (("btwm", "dmc", "walker_walk"), ("dreamerv3", "crafter", "reward")):
        cmd = build_experiment_command(*spec)
        print("command:", " ".join(cmd))


def optional_cpu_debug_train() -> None:
    if os.environ.get("FULL_TRAIN", "0") != "1":
        print("[4/4] Skipping optional CPU debug training smoke (set FULL_TRAIN=1 to enable)")
        return
    print("[4/4] Running optional CPU debug training smoke")
    env = os.environ.copy()
    pythonpath = [str(DREAMERV3_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "dreamerv3" / "dreamerv3" / "main.py"),
            "--logdir",
            str(ROOT / "runs" / "smoke_debug"),
            "--configs",
            "debug",
            "--task",
            "dummy_disc",
            "--run.steps",
            "200",
        ],
        cwd=ROOT,
        check=True,
        env=env,
    )


def main() -> None:
    check_syntax()
    run_unit_tests_if_possible()
    verify_command_generation()
    optional_cpu_debug_train()
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()

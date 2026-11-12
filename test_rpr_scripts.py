import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts" / "rpr"
STAGES = [
    "00_preflight.sh",
    "01_calibrate_rewards.sh",
    "02_prepare_warmup_load.sh",
    "03_dry_run_warmup.sh",
    "04_run_warmup.sh",
    "05_prepare_rpr1_load.sh",
    "06_run_rpr1.sh",
    "07_build_rsr2_data.sh",
    "08_run_rsr2_sft.sh",
    "09_prepare_rpr2_load.sh",
    "10_run_rpr2.sh",
    "11_status.sh",
    "run_all.sh",
]


class RPRScriptTests(unittest.TestCase):
    def test_stage_scripts_are_executable_and_parse(self):
        for name in STAGES:
            path = SCRIPTS / name
            self.assertTrue(path.is_file(), name)
            self.assertTrue(path.stat().st_mode & stat.S_IXUSR, name)
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_prepare_warmup_resolves_tracker_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "sft"
            iteration = checkpoint / "iter_0000007"
            iteration.mkdir(parents=True)
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("7\n")
            work = root / "work"
            env = dict(os.environ, SLIME_PYTHON=sys.executable,
                       SFT_CKPT_ROOT=str(checkpoint), SFT_ACTOR_ITER="",
                       RPR_WORK_ROOT=str(work), NUM_GPUS="4")
            subprocess.run(["bash", str(SCRIPTS / "02_prepare_warmup_load.sh")],
                           cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            load = work / "load_roots" / "warmup"
            self.assertEqual((load / "release").resolve(), iteration.resolve())
            self.assertEqual((load / "latest_checkpointed_iteration.txt").read_text(), "release\n")
            self.assertFalse(Path(str(load) + "_critic").exists())

    def test_status_can_inspect_an_unstarted_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, SLIME_PYTHON=sys.executable,
                       RPR_WORK_ROOT=str(Path(directory) / "work"), NUM_GPUS="4")
            result = subprocess.run(["bash", str(SCRIPTS / "11_status.sh")],
                                    cwd=ROOT, env=env, check=True,
                                    capture_output=True, text=True)
            self.assertIn("warmup     not started", result.stdout)
            self.assertIn("rpr2       not started", result.stdout)


if __name__ == "__main__":
    unittest.main()

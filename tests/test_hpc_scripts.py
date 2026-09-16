"""Exercise the launchers without submitting jobs or installing dependencies."""

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


HPC = Path(__file__).resolve().parents[1] / "hpc"
JOBS = (
    "sbatch_robust_lp_smoke.sh",
    "sbatch_train_robust_lp_agent_cpu.sh",
    "sbatch_robust_lp_seed_sweep_cpu.sh",
    "sbatch_aggregate_domain_randomized_seed_sweep.sh",
)
DEFAULT_RESULTS = "experiments/results/domain_randomized_ppo"


@pytest.fixture
def cluster(tmp_path):
    project = tmp_path / "checkout with spaces"
    (project / "experiments").mkdir(parents=True)
    (project / "experiments/train_robust_lp_agent.py").touch()
    (project / "requirements.txt").touch()
    shutil.copytree(HPC, project / "hpc", ignore=shutil.ignore_patterns(".env"))
    capture = tmp_path / "python_calls.jsonl"
    module_calls = tmp_path / "module_calls.txt"
    scratch = tmp_path / "temporary files"
    scratch.mkdir()

    # Capture Python invocation boundaries, arguments, cwd and runtime settings.
    # The setup script's venv operation is simulated; pip never runs.
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, shlex, shutil, sys\n"
        "keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', "
        "'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED', 'MPLCONFIGDIR')\n"
        "modules = pathlib.Path(os.environ['MODULE_CALLS'])\n"
        "record = {'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'env': {key: os.environ.get(key) for key in keys}, "
        "'modules': modules.read_text().splitlines() if modules.exists() else []}\n"
        "with open(os.environ['CAPTURE'], 'a') as output:\n"
        "    output.write(json.dumps(record) + '\\n')\n"
        "if sys.argv[1:3] == ['-m', 'venv']:\n"
        "    bindir = pathlib.Path(sys.argv[3]) / 'bin'\n"
        "    bindir.mkdir(parents=True)\n"
        "    (bindir / 'activate').write_text('export PATH=' "
        "+ shlex.quote(str(bindir)) + ':\"$PATH\"\\n')\n"
        "    shutil.copy2(__file__, bindir / 'python')\n"
    )
    fake_python.chmod(0o755)

    def make_venv(path):
        bindir = path / "bin"
        bindir.mkdir(parents=True)
        shutil.copy2(fake_python, bindir / "python")
        (bindir / "activate").write_text(
            f'export PATH={shlex.quote(str(bindir))}:"$PATH"\n'
        )

    make_venv(project / "venv")
    env = {
        "PATH": "/usr/bin:/bin",
        "CAPTURE": str(capture),
        "MODULE_CALLS": str(module_calls),
        "TMPDIR": str(scratch),
        "SLURM_SUBMIT_DIR": str(project),
        "SLURM_JOB_ID": "1234",
        "SLURM_ARRAY_JOB_ID": "1234",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_PARTITION": "test_partition",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "16",
    }

    def run(script, overrides=None, *, cwd=None, staged=False):
        invocation_env = env | (overrides or {})
        invocation_env = {k: str(v) for k, v in invocation_env.items() if v is not None}
        source = project / "hpc" / script
        if staged:
            source = tmp_path / "scheduler_copy.sh"
            shutil.copy2(project / "hpc" / script, source)
        result = subprocess.run(
            ["/bin/bash", str(source)],
            cwd=cwd or tmp_path,
            env=invocation_env,
            text=True,
            capture_output=True,
            timeout=15,
        )
        calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        return result, calls

    return project, scratch, make_venv, fake_python, run


def option(call, flag):
    return call["args"][call["args"].index(flag) + 1]


@pytest.mark.parametrize("script", JOBS)
@pytest.mark.parametrize("location", ["submission", "explicit", "working_directory"])
def test_jobs_find_checkout_and_environment_without_modules(cluster, script, location):
    project, scratch, make_venv, _, run = cluster
    overrides = {}
    if location == "explicit":
        custom_venv = project.parent / "external environment"
        make_venv(custom_venv)
        overrides = {
            "SAIFE_PROJECT_DIR": project,
            "SAIFE_VENV_DIR": custom_venv,
            "SLURM_SUBMIT_DIR": project.parent / "not the checkout",
        }
    elif location == "working_directory":
        overrides = {"SLURM_SUBMIT_DIR": None, "SAIFE_VENV_DIR": "venv"}

    # Slurm can execute a copy of the script outside the checkout.
    result, calls = run(script, overrides, cwd=project, staged=True)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    call = calls[0]
    assert call["cwd"] == str(project)
    assert call["modules"] == []
    assert call["env"]["OMP_NUM_THREADS"] == "16"
    assert call["env"]["MKL_NUM_THREADS"] == "16"
    assert call["env"]["OPENBLAS_NUM_THREADS"] == "16"
    assert call["env"]["NUMEXPR_NUM_THREADS"] == "16"
    assert call["env"]["PYTHONUNBUFFERED"] == "1"
    if "aggregate" in script:
        assert call["args"][:2] == ["-u", "experiments/aggregate_domain_randomized_seed_sweep.py"]
        assert option(call, "--input-dir") == f"{DEFAULT_RESULTS}/seed_sweep"
        assert option(call, "--output-dir") == f"{DEFAULT_RESULTS}/seed_sweep/aggregate"
    else:
        assert call["args"][:2] == ["-u", "experiments/train_robust_lp_agent.py"]
        cache = Path(call["env"]["MPLCONFIGDIR"])
        assert cache.parent == scratch
        assert cache.is_dir()
        if "smoke" in script:
            assert "--smoke-test" in call["args"]
            assert option(call, "--output-dir") == f"{DEFAULT_RESULTS}/smoke"
        else:
            suffix = "seed_sweep/seed_43" if "seed_sweep" in script else "full"
            assert option(call, "--output-dir") == f"{DEFAULT_RESULTS}/{suffix}"
            assert option(call, "--seed") == ("43" if "seed_sweep" in script else "42")
            assert option(call, "--total-timesteps") == "10000000"


@pytest.mark.parametrize("script", (*JOBS, "setup_env.sh"))
def test_explicit_module_is_loaded_before_python(cluster, script):
    _, _, _, _, run = cluster
    result, calls = run(script, {
        "SAIFE_PYTHON_MODULE": "python/example",
        "BASH_FUNC_module%%": '() { printf "%s\\n" "$*" >> "$MODULE_CALLS"; }',
    })
    assert result.returncode == 0, result.stderr
    assert calls
    assert all(call["modules"] == ["purge", "load python/example"] for call in calls)


@pytest.mark.parametrize("script", (*JOBS, "setup_env.sh"))
@pytest.mark.parametrize("failure", ["checkout", "environment", "module_unavailable", "module_load"])
def test_missing_prerequisites_stop_before_python(cluster, script, failure):
    project, _, _, _, run = cluster
    if failure == "checkout":
        overrides = {"SAIFE_PROJECT_DIR": project / "missing"}
        expected = "SAIFE_PROJECT_DIR"
    elif failure == "environment":
        invalid_venv = project / "invalid environment"
        invalid_venv.mkdir()
        overrides = {"SAIFE_VENV_DIR": invalid_venv}
        expected = "SAIFE_VENV_DIR"
    elif failure == "module_unavailable":
        overrides = {"SAIFE_PYTHON_MODULE": "python/example"}
        expected = "'module' is unavailable"
    else:
        overrides = {
            "SAIFE_PYTHON_MODULE": "python/example",
            "BASH_FUNC_module%%": '() { if [ "$1" = load ]; then echo "Module load failed" >&2; return 1; fi; }',
        }
        expected = "Module load failed"
    result, calls = run(script, overrides)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not calls


@pytest.mark.parametrize("task_id, seed", [(0, 43), (4, 47), (9, 52)])
def test_sweep_preserves_seed_mapping_and_submission_overrides(cluster, task_id, seed):
    project, _, _, _, run = cluster
    cache = project / "custom cache"
    result, calls = run(JOBS[2], {
        "SLURM_ARRAY_TASK_ID": task_id,
        "OUTPUT_DIR": "results with spaces",
        "MPLCONFIGDIR": cache,
        "TOTAL_TIMESTEPS": "200",
        "NUM_TRAJECTORIES": "2",
        "EVALUATION_SEED": "2468",
        "EVAL_STRESS_SIGMA_VALUES": "0.06 0.09",
    })
    assert result.returncode == 0, result.stderr
    call = calls[0]
    assert option(call, "--seed") == str(seed)
    assert option(call, "--output-dir") == f"results with spaces/seed_{seed}"
    assert option(call, "--total-timesteps") == "200"
    assert option(call, "--num-trajectories") == "2"
    assert option(call, "--train-domains-per-reset") == "2"
    assert option(call, "--evaluation-seed") == "2468"
    start = call["args"].index("--eval-stress-sigma-values") + 1
    assert call["args"][start:start + 2] == ["0.06", "0.09"]
    assert call["env"]["MPLCONFIGDIR"] == str(cache)
    assert cache.is_dir()


def test_smoke_output_and_existing_aggregation_input_can_be_overridden(cluster):
    _, _, _, _, run = cluster
    result, calls = run(JOBS[0], {"OUTPUT_DIR": "custom smoke"})
    assert result.returncode == 0, result.stderr
    assert option(calls[-1], "--output-dir") == "custom smoke"
    result, calls = run(JOBS[3], {"INPUT_DIR": "previous sweep"})
    assert result.returncode == 0, result.stderr
    assert option(calls[-1], "--input-dir") == "previous sweep"
    assert option(calls[-1], "--output-dir") == "previous sweep/aggregate"


def test_setup_creates_environment_using_its_checkout_from_another_directory(cluster):
    project, _, _, fake_python, run = cluster
    fake_python.with_name("python3").symlink_to(fake_python)
    result, calls = run("setup_env.sh", {
        "PATH": f"{fake_python.parent}:/usr/bin:/bin",
        "SAIFE_VENV_DIR": "new environment",
        "SLURM_SUBMIT_DIR": project.parent / "unrelated submission directory",
    })
    assert result.returncode == 0, result.stderr
    assert calls[0]["args"] == ["-m", "venv", str(project / "new environment")]
    assert calls[1]["args"] == ["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    assert calls[2]["args"] == ["-m", "pip", "install", "-r", "requirements.txt"]
    assert calls[3]["args"] == ["-"]
    assert all(call["cwd"] == str(project) for call in calls)
    assert (project / "logs").is_dir()
    assert (project / DEFAULT_RESULTS).is_dir()

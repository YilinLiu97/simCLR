"""
Run multiple SimCLR experiments with different hyperparameters, then plot each one.
Results are saved to plots/<exp_name>/.

Baseline (already trained): temp=0.07, lr=3e-4, bs=32
New experiments vary one hyperparameter at a time.
"""

import os
import subprocess
import sys

PYTHON     = '/mnt/whitsett/yilinliu/miniconda3/envs/simclr/bin/python'
LIB_PATH   = '/mnt/whitsett/yilinliu/miniconda3/envs/simclr/lib'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Env with conda libstdc++ prepended (needed by matplotlib)
plot_env = {**os.environ,
            'LD_LIBRARY_PATH': f"{LIB_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"}

# ── Experiment definitions ──────────────────────────────────────────────────
# Each entry: display name + extra args passed to run.py (on top of defaults)
EXPERIMENTS = [
    {
        'name': 'temp0.05_lr3e-4_bs32',
        'args': ['--temperature', '0.05'],
    },
    {
        'name': 'temp0.10_lr3e-4_bs32',
        'args': ['--temperature', '0.10'],
    },
    {
        'name': 'temp0.20_lr3e-4_bs32',
        'args': ['--temperature', '0.20'],
    },
    {
        'name': 'temp0.07_lr1e-3_bs32',
        'args': ['--lr', '1e-3'],
    },
    {
        'name': 'temp0.07_lr1e-4_bs32',
        'args': ['--lr', '1e-4'],
    },
]

EPOCHS = 200  # keep consistent with baseline


def find_new_run(before: set) -> str:
    """Return the single new directory that appeared in runs/ since `before`."""
    runs_dir = os.path.join(SCRIPT_DIR, 'runs')
    after = set(os.listdir(runs_dir))
    new = after - before
    if not new:
        raise RuntimeError('No new run directory found after training.')
    return os.path.join(runs_dir, sorted(new)[-1])  # latest if somehow >1


def run_training(extra_args: list) -> str:
    """Launch run.py, return path to the newly created run directory."""
    runs_dir = os.path.join(SCRIPT_DIR, 'runs')
    before   = set(os.listdir(runs_dir))

    cmd = [PYTHON, os.path.join(SCRIPT_DIR, 'run.py'),
           '--epochs', str(EPOCHS)] + extra_args
    print(f'  $ {" ".join(cmd)}')
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)

    return find_new_run(before)


def run_plotting(run_dir: str, out_dir: str):
    """Call plot_results.py with conda's libstdc++ available."""
    cmd = [PYTHON, os.path.join(SCRIPT_DIR, 'plot_results.py'),
           '--run_dir', run_dir,
           '--out_dir', out_dir,
           '--epochs',  str(EPOCHS)]
    print(f'  $ {" ".join(cmd)}')
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=plot_env, check=True)


# ── Main loop ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for exp in EXPERIMENTS:
        name = exp['name']
        bar  = '=' * 60
        print(f'\n{bar}')
        print(f'  Experiment: {name}')
        print(f'{bar}')

        # Train
        run_dir = run_training(exp['args'])
        print(f'  Run saved to: {run_dir}')

        # Plot
        out_dir = os.path.join(SCRIPT_DIR, 'plots', name)
        run_plotting(run_dir, out_dir)
        print(f'  Plots saved to: {out_dir}')

    print('\n\nAll experiments done.')
    print('Plot folders:')
    plots_root = os.path.join(SCRIPT_DIR, 'plots')
    for d in sorted(os.listdir(plots_root)):
        print(f'  plots/{d}/')

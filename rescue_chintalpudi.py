"""
RESCUE script for the long-running Chintalpudi FINAL_joint MCMC.

Pulls the in-memory MCMC state out of a *running* python.exe and writes
it to disk WITHOUT killing the process (the process may or may not
survive afterwards — copy the npz file off the machine immediately).

USAGE  (Windows, in a NEW terminal — do NOT touch the MCMC window):

    pip install hypno
    python rescue_chintalpudi.py <PID>

How to find <PID>:

    PowerShell:
        Get-Process python | Format-Table Id, StartTime, CPU
        # pick the python.exe with the oldest StartTime / largest CPU

    Or Task Manager -> Details tab -> sort by CPU time.

Output:
    C:\\Users\\Public\\chintalpudi_partial.npz   (the rescued data)
    C:\\Users\\Public\\chintalpudi_dump.log      (status of the dump)

After the dump, copy chintalpudi_partial.npz somewhere safe.
Then a reconstruction script can turn it into the full 9-plot suite.
"""
import os
import sys
import textwrap


OUT_NPZ = r'C:\Users\Public\chintalpudi_partial.npz'
OUT_LOG = r'C:\Users\Public\chintalpudi_dump.log'


# This is the code that gets injected into the running MCMC process.
# It walks Python frames, finds run_mcmc_3d_rao_joint, and pickles its
# locals to disk. Uses the target process's own numpy (already loaded).
INJECTED = textwrap.dedent(r"""
    import sys, numpy as np, traceback
    OUT = r'__OUT__'
    LOG = r'__LOG__'
    def _log(msg):
        with open(LOG, 'a') as _f:
            _f.write(str(msg) + '\n')
    try:
        target = None
        for _tid, _top in sys._current_frames().items():
            _f = _top
            while _f is not None:
                if _f.f_code.co_name == 'run_mcmc_3d_rao_joint':
                    target = _f
                    break
                _f = _f.f_back
            if target is not None:
                break
        if target is None:
            _log('ERROR: run_mcmc_3d_rao_joint frame not found')
        else:
            L = target.f_locals
            it = L.get('it', -1)
            n_done = (it + 1) if (it is not None and it >= 0) else 0
            am = np.asarray(L.get('all_misfits'))[:n_done] if L.get('all_misfits') is not None else np.array([])
            aa = np.asarray(L.get('all_alphas'))[:n_done]  if L.get('all_alphas')  is not None else np.array([])
            np.savez_compressed(
                OUT,
                iter_done=n_done,
                n_iterations_total=L.get('n_iterations', -1),
                chain=np.asarray(L.get('chain', [])),
                alpha_chain=np.asarray(L.get('alpha_chain', [])),
                all_misfits=am,
                all_alphas=aa,
                current_depths=np.asarray(L.get('current_depths')),
                current_alpha=L.get('current_alpha'),
                current_misfit=L.get('current_misfit'),
                n_accepted=L.get('n_accepted', -1),
                n_depth_proposed=L.get('n_depth_proposed', -1),
                n_depth_accepted=L.get('n_depth_accepted', -1),
                n_alpha_proposed=L.get('n_alpha_proposed', -1),
                n_alpha_accepted=L.get('n_alpha_accepted', -1),
                block_x_edges=np.asarray(L.get('block_x_edges')),
                block_y_edges=np.asarray(L.get('block_y_edges')),
                obs_x=np.asarray(L.get('obs_x')),
                obs_y=np.asarray(L.get('obs_y')),
                gravity_obs=np.asarray(L.get('gravity_obs')),
                drho_0=L.get('drho_0'),
                noise_std=L.get('noise_std'),
                step_depth=L.get('step_depth'),
                step_alpha=L.get('step_alpha'),
                alpha_min=L.get('alpha_min'),
                alpha_max=L.get('alpha_max'),
                alpha_init=L.get('alpha_init'),
                prob_perturb_alpha=L.get('prob_perturb_alpha'),
                smoothness_weight=L.get('smoothness_weight'),
                n_sublayers=L.get('n_sublayers'),
                depth_min=L.get('depth_min'),
                depth_max=L.get('depth_max'),
                seed=L.get('seed'),
            )
            _log('OK: iter_done=%d  chain_len=%d  alpha_len=%d  ->  %s' % (
                n_done, len(L.get('chain', [])), len(L.get('alpha_chain', [])), OUT))
    except Exception:
        _log('EXC:\n' + traceback.format_exc())
""").replace('__OUT__', OUT_NPZ).replace('__LOG__', OUT_LOG)


def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print(__doc__)
        sys.exit(1)
    pid = int(sys.argv[1])

    try:
        from hypno import inject_py
    except ImportError:
        print("hypno not installed. Run:  pip install hypno")
        sys.exit(1)

    # Wipe old log so we only see this run's output
    try:
        if os.path.exists(OUT_LOG):
            os.remove(OUT_LOG)
    except OSError:
        pass

    print(f"Injecting dump code into PID {pid} ...")
    inject_py(pid, INJECTED)
    print("Injection call returned. Check the log:")
    print(f"  {OUT_LOG}")
    print(f"And the rescued data:")
    print(f"  {OUT_NPZ}")
    print()
    print("If you see 'OK: iter_done=...' in the log, COPY the npz off the")
    print("machine immediately (USB / OneDrive / scp). The MCMC process may")
    print("or may not survive — your data is what matters now.")


if __name__ == '__main__':
    main()

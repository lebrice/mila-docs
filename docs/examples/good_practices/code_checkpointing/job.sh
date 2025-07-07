
set -eof pipefail

# Expect this variable to be set by the `safe_sbatch` submission script or similar.


# Prevents future changes in the python files from messing up future jobs.
repo="$HOME/repos/scaling_pqn" # could also use the current directory.
dest="$SLURM_TMPDIR/$(basename "$repo")"
if [[ -n "$GIT_COMMIT" ]]; then
    git clone "$repo" "$dest"
    echo "Checking out commit $GIT_COMMIT"
    cd "$dest"
    git checkout $GIT_COMMIT
elif [[ -n "$(git -C $repo status --porcelain)" ]]; then
    echo "Warning: GIT_COMMIT is not set and the current repo at ~/repos/scaling_pqn has uncommitted changes."
    echo "This may cause future jobs to fail or produce inconsistent results!"
    echo "Consider using the 'safe_sbatch' script to submit jobs instead."
else
    echo "GIT_COMMIT environment variable is not set, but the repo state is clean. "
    echo "If you modify the files in the repo, future jobs might fail or produce inconsistent results. "
fi

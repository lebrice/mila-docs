#!/bin/bash
# Use this to update the diffs based on the contents of the files.

set -e

generate_diff() {
    echo "Generating diff for docs/examples/$1 -> docs/examples/$2"
    # NOTE: Assuming that this gets run from the `docs` folder (as is the case when building the docs).

    # TODO: Create a diff file that can then be used to make writing new examples easier.
    git diff --no-index \
        "examples/$1" \
        "examples/$2" \
        | grep -Ev "^index" \
        > examples/$2.diff

    # Write a diff file to be shown in the documentation.
    mkdir -p $(dirname "_build/examples/$2.diff")
    git diff --no-index -U9999 \
        "examples/$1" \
        "examples/$2" \
        | grep -Ev "^--- |^\+\+\+ |^@@ |^index |^diff --git " \
        > _build/examples/$2.diff
}

# single_gpu -> multi_gpu
generate_diff distributed/001_single_gpu/job.sh distributed/002_multi_gpu/job.sh
generate_diff distributed/001_single_gpu/main.py distributed/002_multi_gpu/main.py

# multi_gpu -> multi_node
generate_diff distributed/002_multi_gpu/job.sh distributed/003_multi_node/job.sh
generate_diff distributed/002_multi_gpu/main.py distributed/003_multi_node/main.py
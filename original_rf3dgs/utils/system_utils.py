#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    """Create a directory including any missing parents, ignoring 'already exists'.

    Equivalent to `mkdir -p` on the command line. Predates os.makedirs(exist_ok=True),
    which is what modern code would use instead.
    """
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        # EEXIST is only benign when the thing in the way is itself a directory:
        # if a *file* already occupies that path we must not swallow the error.
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    """Return the highest iteration number among the checkpoints saved in `folder`.

    Used to resolve `--iteration -1` (meaning "the latest") into a concrete number.
    Expects entries named like 'iteration_30000'; the number is taken from the
    text after the final underscore.
    """
    # Every entry is parsed, so a stray file in this folder whose name does not end
    # in '_<number>' will raise ValueError rather than being skipped.
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    # An empty folder raises ValueError from max() on an empty sequence.
    return max(saved_iters)

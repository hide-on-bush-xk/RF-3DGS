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

# All command-line configuration for train.py / render.py / the viewer.
#
# The pattern: each ParamGroup subclass declares its options as plain attributes
# with default values in __init__. The base class reflects over those attributes to
# register argparse flags automatically, so adding an option means adding one line
# here and nothing else. A leading underscore on the attribute name additionally
# creates a single-letter short flag (_source_path -> --source_path / -s).

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    """Empty container that ParamGroup.extract() fills with one group's values."""
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        """Register every attribute of `self` as a command-line flag.

        fill_none (passed as `sentinel` by ModelParams) replaces every default with
        None. That is how render.py distinguishes "the user typed this flag" from
        "this flag was left alone": with None defaults, get_combined_args can let
        the saved cfg_args file supply the untouched values.
        """
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                # Underscore prefix is the marker for "also give this a short flag".
                shorthand = True
                key = key[1:]
            t = type(value)               # the declared default's type drives argparse's type=
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    # Short flag is just the first letter, e.g. -w for --white_background.
                    # store_true means bool options can only be switched on, never off.
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        """Pull just this group's options out of the full parsed namespace.

        Matches both the public name and the underscored one, so `source_path` in
        args is recognised as belonging to the group that declared `_source_path`.
        """
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    """What to load and where to put results. Flags: -s -m -i -r -w."""
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3              # max SH band; 3 -> 16 coefficients per channel
        self._source_path = ""          # -s, the dataset directory
        self._model_path = ""           # -m, the output directory
        self._images = "images"         # -i, image subfolder name inside a COLMAP scene
        self._resolution = -1           # -r, see utils/camera_utils.loadCam for the encoding
        self._white_background = False  # -w, white instead of black where nothing is drawn
        self.data_device = "cuda"       # where ground-truth images live; "cpu" saves VRAM
        self.eval = False               # hold out a test split instead of training on everything
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        """As ParamGroup.extract, but also absolutises source_path.

        Careful: os.path.abspath("") returns the current working directory. So
        omitting -s does not raise -- it silently resolves to wherever you launched
        the script, and the failure surfaces much later as
        "Could not recognize scene type!" from scene/__init__.py.
        """
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    """Switches that move work between the CUDA rasteriser and Python.

    Both *_python flags exist for debugging and for the interactive viewer, which
    toggles them live. The defaults (CUDA does everything) are what you want for
    training speed.
    """
    def __init__(self, parser):
        self.convert_SHs_python = False    # evaluate spherical harmonics in Python
        self.compute_cov3D_python = False  # build 3D covariances in Python
        self.debug = False                 # verbose rasteriser diagnostics
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    """Learning rates and the adaptive density control schedule.

    The defaults are the published 3DGS settings, tuned for 30k iterations on
    photographic scenes. RF-3DGS training typically runs 40k from a checkpoint,
    with geometry frozen, so the position/scaling/rotation rates below have no
    effect in that stage.
    """
    def __init__(self, parser):
        self.iterations = 30_000
        # Positions use an exponential decay from init to final over max_steps;
        # the delay_mult warms it up from 1% of the initial rate. Every other rate
        # below is constant.
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025        # SH coefficients (features_rest gets this / 20)
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        # A Gaussian counts as "large" when its extent exceeds this fraction of the
        # scene radius; large ones are split, small ones cloned.
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2         # weight of the (1 - SSIM) term in the loss
        self.densification_interval = 100    # run densify_and_prune every N iterations
        self.opacity_reset_interval = 3000   # push all opacities back to near zero every N
        self.densify_from_iter = 500         # no densification before this
        self.densify_until_iter = 15_000     # Gaussian count is frozen after this
        # Screen-space position-gradient threshold above which a Gaussian is
        # cloned or split. The single most influential knob on final point count.
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    """Merge command-line arguments with the cfg_args file saved next to the model.

    This is why render.py and the viewer need only -m: everything else (source_path,
    eval, white_background, sh_degree, ...) is recovered from the training run. The
    command line wins wherever it specified something.
    """
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"      # fallback: an empty namespace
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        # Only catches model_path being None (os.path.join then raises TypeError).
        # A model_path that exists but has no cfg_args raises FileNotFoundError,
        # which is *not* caught here and will terminate the script.
        print("Config file not found at")
        pass
    # The file holds a Namespace repr, so it is revived with eval(). That executes
    # whatever the file contains -- fine for your own output directories, worth
    # knowing before pointing -m at a model directory from an untrusted source.
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        # Only non-None command-line values override the file. This is exactly what
        # the sentinel=True defaults above are for: untouched flags are None and so
        # fall through to the saved configuration.
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

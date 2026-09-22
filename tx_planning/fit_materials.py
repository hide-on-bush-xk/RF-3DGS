"""Stage 3: learn the materials from one transmitter, predict from another.

A planner's digital twin is only useful if it predicts positions it was not
measured from. This fits every radio material's scattering coefficient to the
power delay profiles observed on the indoor grid from transmitter A (synthetic
"measurements": the same scene with hidden per-material truth values), then
scores the fitted twin against the truth from transmitter B, where nothing
was fitted. The baseline is the un-fitted twin with every material at the
initial value.

Gradients come from Sionna's differentiable solver, as verified in
probe_differentiability.py. The observation per receiver is a delay-binned
power delay profile in dB: total power alone barely depends on how a wall
splits its reflection between specular and diffuse, but the PDP does -- the
specular part is a spike, the diffuse part a tail behind it. `--pdp-bins 1`
falls back to total received power.

`MaterialTwin` is the reusable part; active_measurement.py builds on it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from scene_common import (LOBBY_X, LOBBY_Y, RX_HEIGHT, enable_reverse_mode,
                          indoor_mask, load_radio_scene, make_solver, rx_grid)


def add_twin_args(ap):
    """Register the arguments MaterialTwin reads. Shared with active_measurement.py."""
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--rx-step", type=float, default=2.0)
    ap.add_argument("--truth-seed", type=int, default=0,
                    help="hidden per-material truth drawn in [0.2, 0.9]")
    ap.add_argument("--init", type=float, default=0.5)
    ap.add_argument("--obs-noise-db", type=float, default=1.0,
                    help="Gaussian noise added to the observed PDPs")
    ap.add_argument("--noise-floor-db", type=float, default=-130.0)
    ap.add_argument("--pdp-bins", type=int, default=16)
    ap.add_argument("--pdp-bin-ns", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.05,
                    help="largest per-step move of any coefficient")
    ap.add_argument("--optimizer", choices=["ngd", "adam", "adam-rel"], default="ngd",
                    help="ngd scales the whole gradient by its largest entry, "
                         "so a material the data barely constrains barely moves "
                         "(identifiable-material error 0.025 on the NIST lobby); "
                         "adam moves every material at the same rate, and the "
                         "unconstrained ones wander off; adam-rel is adam whose "
                         "epsilon is --eps-rel of the largest running gradient "
                         "-- it keeps the unconstrained ones still but its "
                         "sign-like steps overshoot on noisy gradients (0.051)")
    ap.add_argument("--eps-rel", type=float, default=0.05)
    ap.add_argument("--samples", type=int, default=30_000)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")


class MaterialTwin:
    """The scene with one transmitter, the indoor receiver grid, and every
    used material's scattering coefficient as a fittable parameter."""

    def __init__(self, cfg):
        import drjit as dr
        import mitsuba as mi
        mi.set_variant(cfg.variant)
        enable_reverse_mode()
        from sionna.rt import Receiver, Transmitter
        self.dr, self.mi, self.cfg = dr, mi, cfg

        self.scene = load_radio_scene(cfg.scene_xml)
        self.solver = make_solver(reverse_mode=True)
        rx = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, cfg.rx_step)
        self.rx_positions = rx[indoor_mask(self.scene, rx)]
        # Receivers are fixed for the lifetime of the twin; only the transmitter
        # and the material values move.
        for i, p in enumerate(self.rx_positions):
            self.scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
        self.tx = Transmitter(name="tx", position=[0.0, 0.0, 2.0])
        self.scene.add(self.tx)
        self.noise = 10.0 ** (cfg.noise_floor_db / 10.0)

        # Only materials attached to some object can be fitted or observed.
        self.mats = {n: m for n, m in self.scene.radio_materials.items() if m.is_used}
        self.names = sorted(self.mats)        # sorted, so orderings are stable
        self.rng = np.random.default_rng(cfg.truth_seed)
        # The hidden ground truth. It is drawn here and never read by the fit --
        # only by the scoring, which is what makes this a proper recovery test.
        self.truth = dict(zip(self.names,
                              self.rng.uniform(0.2, 0.9, len(self.names)).round(3).tolist()))

    # -- scene state ------------------------------------------------------
    def set_materials(self, values):
        """Write a {name: coefficient} dict into the live scene."""
        for n in self.names:
            self.mats[n].scattering_coefficient = float(values[n])

    def set_tx(self, pos):
        """Move the single transmitter."""
        self.tx.position = [float(v) for v in pos]

    # -- forward model ----------------------------------------------------
    def observe(self):
        """Per-bin received power in dB: a list of `pdp_bins` tensors [num_rx].

        Delays do not depend on the materials, so the bin masks are constants
        and the sum inside each bin stays differentiable.
        """
        dr, mi, cfg = self.dr, self.mi, self.cfg
        paths = self.solver(scene=self.scene, max_depth=cfg.max_depth,
                            samples_per_src=cfg.samples, los=True,
                            specular_reflection=True, diffuse_reflection=True,
                            refraction=False, synthetic_array=True, seed=42)
        a_re, a_im = paths.a                                  # [rx, 1, 1, 1, P]
        if a_re.shape[-1] == 0:
            raise RuntimeError(f"no paths from transmitter at "
                               f"{np.asarray(self.tx.position).reshape(-1)}: "
                               "it is inside an object or outside the building")
        p = dr.square(a_re) + dr.square(a_im)
        # Collapse the three singleton antenna/tx axes, keeping [rx, paths].
        for axis in (3, 2, 1):
            p = dr.sum(p, axis=axis)                          # [rx, P]
        # tau and valid are taken as plain numpy: they are the constants the
        # docstring refers to, so no gradient flows through the binning itself.
        tau = np.asarray(paths.tau)[:, 0, :]
        valid = np.asarray(paths.valid)[:, 0, :] & (tau >= 0)
        b = np.floor(tau / (cfg.pdp_bin_ns * 1e-9)).astype(int)
        # Everything past the last bin is folded INTO it; invalid paths get -1,
        # which matches no bin and so is dropped.
        b = np.where(valid, np.minimum(b, cfg.pdp_bins - 1), -1)
        out = []
        for k in range(cfg.pdp_bins):
            # Multiplying by a 0/1 constant mask keeps the sum differentiable in
            # the path amplitudes while selecting only this bin's paths.
            mask = mi.TensorXf((b == k).astype(np.float32))
            p_k = dr.sum(p * mask, axis=1)
            out.append(10.0 * dr.log(p_k + self.noise) / math.log(10.0))
        return out

    @staticmethod
    def as_np(t):
        """Dr.Jit tensor -> flat numpy; a list of them -> a [bins, rx] array."""
        if isinstance(t, list):
            return np.stack([MaterialTwin.as_np(x) for x in t])   # [bins, rx]
        return np.asarray(t).reshape(-1).astype(float)

    def measure(self, tx_pos, values, noisy=True):
        """Synthetic measurement from `tx_pos` with the given materials.

        noisy=True stands in for a real measurement; noisy=False produces the
        clean reference a fit is scored against.
        """
        self.set_tx(tx_pos)
        self.set_materials(values)
        obs = self.as_np(self.observe())
        if noisy and self.cfg.obs_noise_db > 0:
            obs = obs + self.rng.normal(0, self.cfg.obs_noise_db, obs.shape)
        return obs

    def rmse(self, values, tx_pos, ref):
        """RMSE in dB between this twin's prediction at tx_pos and `ref`."""
        self.set_tx(tx_pos)
        self.set_materials(values)
        return float(np.sqrt(np.mean((self.as_np(self.observe()) - ref) ** 2)))

    # -- gradients --------------------------------------------------------
    def _params_with_grad(self, values):
        """Set the materials and mark every coefficient differentiable.

        Must be called fresh before each backward pass: Dr.Jit's gradient state
        does not survive the scene write in set_materials.
        """
        self.set_materials(values)
        params = {n: self.mats[n].scattering_coefficient for n in self.names}
        for p in params.values():
            self.dr.enable_grad(p)
        return params

    def _grads(self, params):
        """Read the accumulated gradients back as plain floats."""
        return {n: float(self.as_np(self.dr.grad(params[n]))[0]) for n in self.names}

    def loss_and_grad(self, values, tx_pos, obs):
        """Mean squared dB error of the PDPs at one transmitter, and d/d s_n."""
        dr, mi = self.dr, self.mi
        self.set_tx(tx_pos)
        params = self._params_with_grad(values)
        pred = self.observe()
        # Averaged over bins as well as receivers, so --pdp-bins does not change
        # the loss scale and the same --lr works at any binning.
        loss = sum(dr.mean(dr.square(pred[k] - mi.TensorXf(obs[k].astype(np.float32))))
                   for k in range(self.cfg.pdp_bins)) / self.cfg.pdp_bins
        dr.backward(loss)
        return float(self.as_np(loss)[0]), self._grads(params)

    def sensitivity(self, values, tx_pos, n_probes=4, seed=0):
        """||d PDP / d s_n||^2 per material, without observing anything.

        Hutchinson-style: for Gaussian w, E[(J_n . w)^2] = ||J_n||^2, so a few
        backward passes through random projections of the PDP estimate how
        strongly each material shows in what this transmitter would measure.
        """
        dr, mi = self.dr, self.mi
        rng = np.random.default_rng(seed)
        self.set_tx(tx_pos)
        acc = {n: 0.0 for n in self.names}
        for _ in range(n_probes):
            params = self._params_with_grad(values)
            pred = self.observe()
            # One Gaussian probe vector over receivers, shared across bins.
            proj = sum(dr.sum(pred[k] * mi.TensorXf(
                rng.normal(size=len(self.rx_positions)).astype(np.float32)))
                for k in range(self.cfg.pdp_bins))
            dr.backward(proj)
            # Averaged over probes: this is the expectation the estimator needs.
            for n, g in self._grads(params).items():
                acc[n] += g * g / n_probes
        return acc

    # -- fitting ----------------------------------------------------------
    def fit(self, observations, verbose=True):
        """Fit the materials to {tx_pos: obs} pairs; returns (est, history)."""
        cfg = self.cfg
        est = {n: cfg.init for n in self.names}
        m = {n: 0.0 for n in self.names}; v = {n: 0.0 for n in self.names}
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        for step in range(1, cfg.steps + 1):
            loss_total = 0.0
            grads = {n: 0.0 for n in self.names}
            # Several transmitters are averaged, not concatenated, so adding an
            # observation does not change the loss scale.
            for tx_pos, obs in observations:
                loss, g = self.loss_and_grad(est, tx_pos, obs)
                loss_total += loss / len(observations)
                for n in self.names:
                    grads[n] += g[n] / len(observations)
            # The gradient is recorded every step: identifiable() reads it back.
            history.append({"step": step, "loss_db2": loss_total,
                            "est": dict(est), "grad": grads})
            if verbose:
                print(f"  step {step:3d}  RMSE {math.sqrt(loss_total):5.2f} dB  "
                      f"mean|est-truth| "
                      f"{np.mean([abs(est[n]-self.truth[n]) for n in self.names]):.3f}")
            # `or 1.0` catches an all-zero gradient, which would divide by zero.
            g_max = max(abs(g) for g in grads.values() if math.isfinite(g)) or 1.0
            for n in self.names:
                g = grads[n]
                # A non-finite gradient for one material is skipped rather than
                # poisoning the whole step.
                if not math.isfinite(g):
                    continue
                m[n] = b1 * m[n] + (1 - b1) * g
                v[n] = b2 * v[n] + (1 - b2) * g * g
            rms_max = max(math.sqrt(v[n] / (1 - b2 ** step)) for n in self.names) or 1.0
            for n in self.names:
                if not math.isfinite(grads[n]):
                    continue
                mh, vh = m[n] / (1 - b1 ** step), v[n] / (1 - b2 ** step)
                if cfg.optimizer == "adam":
                    # Per-parameter normalisation: every material moves at ~lr,
                    # including ones the data says nothing about.
                    upd = cfg.lr * mh / (math.sqrt(vh) + eps)
                elif cfg.optimizer == "adam-rel":
                    # Epsilon tied to the largest running gradient, so a material
                    # with a tiny gradient is damped instead of normalised up.
                    upd = cfg.lr * mh / (math.sqrt(vh) + cfg.eps_rel * rms_max)
                else:
                    # ngd (the default): one global scale. Relative gradient
                    # magnitudes survive, so unconstrained materials stay put.
                    upd = cfg.lr * grads[n] / g_max
                # Minus: descent on the loss. Clipped to a physical range.
                est[n] = float(np.clip(est[n] - upd, 0.01, 0.99))
        return est, history

    def identifiable(self, history, frac=0.02):
        """Materials whose mean |gradient| over the fit is >= frac of the largest.

        A material below this threshold was never really constrained by the data,
        so its fitted value is meaningless and reporting an error for it would
        flatter or damn the fit for no reason.
        """
        g_abs = {n: float(np.mean([abs(h["grad"][n]) for h in history])) for n in self.names}
        top = max(g_abs.values())
        return [n for n in self.names if g_abs[n] >= frac * top], g_abs

    def material_error(self, est, subset=None):
        """Mean |estimate - truth|, over `subset` or over every material."""
        subset = subset or self.names
        return float(np.mean([abs(est[n] - self.truth[n]) for n in subset]))


def main():
    """Fit at Tx A, score at held-out Tx B, and report per-material recovery."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_twin_args(ap)
    ap.add_argument("--out", default="../output/tx_planning/fit_materials.json")
    ap.add_argument("--tx-a", type=float, nargs=3, default=[8.0, -7.0, 2.0],
                    help="transmitter the observations come from")
    ap.add_argument("--tx-b", type=float, nargs=3, default=[0.0, -3.0, 2.0],
                    help="held-out transmitter the fitted twin is scored at")
    cfg = ap.parse_args()

    twin = MaterialTwin(cfg)
    print(f"{len(twin.rx_positions)} indoor receivers, {len(twin.names)} used materials")
    t0 = time.time()
    obs_a = twin.measure(cfg.tx_a, twin.truth)            # noisy, the "measurement"
    true_b = twin.measure(cfg.tx_b, twin.truth, noisy=False)   # clean, the target

    def score(values, label):
        """RMSE at both transmitters. Tx A shows fit quality, Tx B generalisation."""
        e_a = twin.rmse(values, cfg.tx_a, obs_a)
        e_b = twin.rmse(values, cfg.tx_b, true_b)
        print(f"{label:28s} RMSE at Tx A {e_a:5.2f} dB   at held-out Tx B {e_b:5.2f} dB")
        return e_a, e_b

    init = {n: cfg.init for n in twin.names}
    base_a, base_b = score(init, "baseline (all materials init)")
    est, history = twin.fit([(cfg.tx_a, obs_a)])
    fit_a, fit_b = score(est, "fitted at Tx A")

    ident, g_abs = twin.identifiable(history)
    print(f"\n{len(ident)}/{len(twin.names)} materials identifiable at Tx A "
          f"(mean |grad| >= 2% of the largest); their mean |est - truth|: "
          f"init {twin.material_error(init, ident):.3f} -> "
          f"fitted {twin.material_error(est, ident):.3f}")
    # Sorted by how strongly the data constrained each material, so the ones
    # whose fitted values mean something appear first; * marks identifiable.
    print("material               truth   init  fitted   mean|grad|")
    for n in sorted(twin.names, key=lambda n: -g_abs[n]):
        print(f"  {n[:22]:22s} {twin.truth[n]:.3f}  {cfg.init:.3f}  {est[n]:.3f}   "
              f"{g_abs[n]:.2e}{'  *' if n in ident else ''}")

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w", encoding="utf-8") as fid:
        json.dump({"config": vars(cfg), "materials": twin.names, "truth": twin.truth,
                   "fitted": est, "identifiable": ident, "history": history,
                   "rmse_db": {"baseline_a": base_a, "baseline_b": base_b,
                               "fitted_a": fit_a, "fitted_b": fit_b},
                   "seconds": time.time() - t0}, fid, indent=1)
    # The headline claim of stage 3: the held-out transmitter's error must fall.
    print(f"\nheld-out Tx B: baseline {base_b:.2f} dB -> fitted {fit_b:.2f} dB "
          f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()

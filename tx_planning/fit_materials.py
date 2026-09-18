"""Stage 3: learn the materials from one transmitter, predict from another.

A planner's digital twin is only useful if it predicts positions it was not
measured from. This fits every radio material's scattering coefficient to the
received power observed on the indoor grid from transmitter A (synthetic
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--out", default="../output/tx_planning/fit_materials.json")
    ap.add_argument("--tx-a", type=float, nargs=3, default=[8.0, -7.0, 2.0],
                    help="transmitter the observations come from")
    ap.add_argument("--tx-b", type=float, nargs=3, default=[0.0, -3.0, 2.0],
                    help="held-out transmitter the fitted twin is scored at")
    ap.add_argument("--rx-step", type=float, default=2.0)
    ap.add_argument("--truth-seed", type=int, default=0,
                    help="hidden per-material truth drawn in [0.2, 0.9]")
    ap.add_argument("--init", type=float, default=0.5)
    ap.add_argument("--obs-noise-db", type=float, default=1.0,
                    help="Gaussian noise added to the observed gains")
    ap.add_argument("--noise-floor-db", type=float, default=-130.0)
    ap.add_argument("--pdp-bins", type=int, default=16)
    ap.add_argument("--pdp-bin-ns", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05,
                    help="largest per-step move of any coefficient")
    ap.add_argument("--optimizer", choices=["ngd", "adam"], default="ngd",
                    help="ngd scales the whole gradient by its largest entry, "
                         "so a material the data barely constrains barely moves; "
                         "adam moves every material at the same rate, and the "
                         "unconstrained ones wander off")
    ap.add_argument("--samples", type=int, default=50_000)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    cfg = ap.parse_args()

    import drjit as dr
    import mitsuba as mi
    mi.set_variant(cfg.variant)
    enable_reverse_mode()
    from sionna.rt import Receiver, Transmitter

    scene = load_radio_scene(cfg.scene_xml)
    solver = make_solver(reverse_mode=True)
    rx_positions = rx_grid(LOBBY_X, LOBBY_Y, RX_HEIGHT, cfg.rx_step)
    rx_positions = rx_positions[indoor_mask(scene, rx_positions)]
    for i, p in enumerate(rx_positions):
        scene.add(Receiver(name=f"rx{i}", position=[float(v) for v in p]))
    tx = Transmitter(name="tx", position=[float(v) for v in cfg.tx_a])
    scene.add(tx)
    noise = 10.0 ** (cfg.noise_floor_db / 10.0)

    # Only materials attached to some object can be fitted or observed.
    mats = {name: m for name, m in scene.radio_materials.items() if m.is_used}
    names = sorted(mats)
    rng = np.random.default_rng(cfg.truth_seed)
    truth = dict(zip(names, rng.uniform(0.2, 0.9, len(names)).round(3).tolist()))
    print(f"{len(rx_positions)} indoor receivers, {len(names)} used materials")
    for n in names:
        print(f"  {n:24s} truth {truth[n]:.3f}")

    def set_materials(values):
        for n in names:
            mats[n].scattering_coefficient = float(values[n])

    def gains_db():
        """Per-bin received power in dB, a list of `pdp_bins` tensors [num_rx].

        Delays do not depend on the materials, so the bin masks are constants
        and the sum inside each bin stays differentiable.
        """
        paths = solver(scene=scene, max_depth=cfg.max_depth,
                       samples_per_src=cfg.samples, los=True,
                       specular_reflection=True, diffuse_reflection=True,
                       refraction=False, synthetic_array=True, seed=42)
        a_re, a_im = paths.a                              # [rx, 1, 1, 1, P]
        p = dr.square(a_re) + dr.square(a_im)
        for axis in (3, 2, 1):
            p = dr.sum(p, axis=axis)                      # [rx, P]
        tau = np.asarray(paths.tau)[:, 0, :]
        valid = np.asarray(paths.valid)[:, 0, :] & (tau >= 0)
        b = np.floor(tau / (cfg.pdp_bin_ns * 1e-9)).astype(int)
        b = np.where(valid, np.minimum(b, cfg.pdp_bins - 1), -1)
        out = []
        for k in range(cfg.pdp_bins):
            mask = mi.TensorXf((b == k).astype(np.float32))
            p_k = dr.sum(p * mask, axis=1)
            out.append(10.0 * dr.log(p_k + noise) / math.log(10.0))
        return out

    def as_np(t):
        if isinstance(t, list):
            return np.stack([as_np(x) for x in t])         # [bins, rx]
        return np.asarray(t).reshape(-1).astype(float)

    # --- synthetic measurements from the hidden truth ----------------------
    set_materials(truth)
    obs_a = as_np(gains_db())
    obs_a += rng.normal(0, cfg.obs_noise_db, obs_a.shape)
    tx.position = [float(v) for v in cfg.tx_b]
    true_b = as_np(gains_db())
    tx.position = [float(v) for v in cfg.tx_a]

    def rmse(pred, ref):
        return float(np.sqrt(np.mean((pred - ref) ** 2)))

    def score(values, label):
        set_materials(values)
        e_a = rmse(as_np(gains_db()), obs_a)
        tx.position = [float(v) for v in cfg.tx_b]
        e_b = rmse(as_np(gains_db()), true_b)
        tx.position = [float(v) for v in cfg.tx_a]
        print(f"{label:28s} RMSE at Tx A {e_a:5.2f} dB   at held-out Tx B {e_b:5.2f} dB")
        return e_a, e_b

    est = {n: cfg.init for n in names}
    base_a, base_b = score(est, "baseline (all materials init)")

    # --- fit at Tx A -------------------------------------------------------
    m = {n: 0.0 for n in names}; v = {n: 0.0 for n in names}
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        set_materials(est)
        params = {n: mats[n].scattering_coefficient for n in names}
        for p in params.values():
            dr.enable_grad(p)
        pred = gains_db()
        loss = sum(dr.mean(dr.square(pred[k] - mi.TensorXf(obs_a[k].astype(np.float32))))
                   for k in range(cfg.pdp_bins)) / cfg.pdp_bins
        dr.backward(loss)
        grads = {n: float(as_np(dr.grad(params[n]))[0]) for n in names}
        loss_v = float(as_np(loss)[0])
        history.append({"step": step, "loss_db2": loss_v, "est": dict(est),
                        "grad": grads})
        print(f"  step {step:3d}  RMSE_A {math.sqrt(loss_v):5.2f} dB  "
              f"mean|est-truth| {np.mean([abs(est[n]-truth[n]) for n in names]):.3f}")
        g_max = max(abs(g) for g in grads.values() if math.isfinite(g)) or 1.0
        for n in names:                                   # descent
            g = grads[n]
            if not math.isfinite(g):
                continue
            if cfg.optimizer == "adam":
                m[n] = b1 * m[n] + (1 - b1) * g
                v[n] = b2 * v[n] + (1 - b2) * g * g
                mh, vh = m[n] / (1 - b1 ** step), v[n] / (1 - b2 ** step)
                upd = cfg.lr * mh / (math.sqrt(vh) + eps)
            else:
                upd = cfg.lr * g / g_max
            est[n] = float(np.clip(est[n] - upd, 0.01, 0.99))

    fit_a, fit_b = score(est, "fitted at Tx A")
    # A material is identifiable at Tx A when its gradient is not negligible
    # next to the largest one, averaged over the fit.
    g_abs = {n: np.mean([abs(h["grad"][n]) for h in history]) for n in names}
    g_top = max(g_abs.values())
    ident = [n for n in names if g_abs[n] >= 0.02 * g_top]
    err0 = np.mean([abs(cfg.init - truth[n]) for n in ident])
    err1 = np.mean([abs(est[n] - truth[n]) for n in ident])
    print(f"\n{len(ident)}/{len(names)} materials identifiable at Tx A "
          f"(mean |grad| >= 2% of the largest); their mean |est - truth|: "
          f"init {err0:.3f} -> fitted {err1:.3f}")
    print("material               truth   init  fitted   mean|grad|")
    for n in sorted(names, key=lambda n: -g_abs[n]):
        print(f"  {n[:22]:22s} {truth[n]:.3f}  {cfg.init:.3f}  {est[n]:.3f}   "
              f"{g_abs[n]:.2e}{'  *' if n in ident else ''}")

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w", encoding="utf-8") as fid:
        json.dump({"config": vars(cfg), "materials": names, "truth": truth,
                   "fitted": est, "identifiable": ident, "history": history,
                   "rmse_db": {"baseline_a": base_a, "baseline_b": base_b,
                               "fitted_a": fit_a, "fitted_b": fit_b},
                   "seconds": time.time() - t0}, fid, indent=1)
    print(f"\nheld-out Tx B: baseline {base_b:.2f} dB -> fitted {fit_b:.2f} dB "
          f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()

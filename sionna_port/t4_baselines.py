"""T4 baselines for the held-out views of an MVDR Sionna RT dataset (docs/t4_channel_model_plan.md).

Writes, for every held-out view, a native-dB spectrum in the layout of a trained run (<out>/renders/<view>.npy), so
rrf_gsplat/mvdr_peaks.py scores it exactly as it scores the radio radiance fields:

  inh_open_r<k>, inh_mixed_r<k>   3GPP TR 38.901 v19.2 indoor office (Sionna 2.1 InH, system level), realisation k:
                                  the CIR at the view's receive array (the ray tracer's 10 x 10 UPA, lambda / 2,
                                  38.901 element, V pol, the face's yaw) from the dataset's transmitter (1 x 1,
                                  38.901, V); path loss and shadow fading on; LOS direction from the geometry;
                                  spatial consistency on, so a position's four faces see one channel
  los                             one free-space direct path (Friis at the carrier, the same element patterns)
  nn                              the nearest training position's truth spectrum, same face
then the same merge_paths_to_time_grid (0.1 ns taps) + mvdr_spectrum as the truth. A 38.901 cluster's rays share
a delay, so InH's (and LOS's) delay-tap covariance is rank-deficient for 100 elements: diagonal loading
(--loading, of tr(R) / M^2) is applied to them -- the truth used none.

--control runs the analytic check first: a single ray from a known direction through the same response ->
MVDR path must peak at that direction (the element ordering and the phase convention of Sionna's PanelArray
against rf_spectra's steering vectors).

    python sionna_port/t4_baselines.py --truth RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct --control
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

C0 = 299792458.0


def read_views(truth):
    """name -> (rx position [3], yaw) for every view of the dataset, and the train / test name lists."""
    from generate_dataset import VIEW_YAWS, euler_to_quaternion
    yaw_R = {y: euler_to_quaternion([y, 0.0, 0.0])[0].as_matrix() for y in VIEW_YAWS}

    def rotmat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    views = {}
    for line in open(os.path.join(truth, "sparse", "0", "images.txt")):
        p = line.split()
        if len(p) < 10 or not p[9].lower().endswith(".png"):
            continue
        R = rotmat([float(v) for v in p[1:5]]); t = np.array([float(v) for v in p[5:8]])
        views[p[9][:-4]] = ((-R.T @ t), min(VIEW_YAWS, key=lambda y: np.abs(yaw_R[y] - R).sum()))
    idx = lambda f: [l.strip() for l in open(os.path.join(truth, f)) if l.strip()]            # noqa: E731
    return views, idx("train_index.txt"), idx("test_index.txt")


def permutation(M, lam, ant_pos):
    """Index into PanelArray's element axis for each of rf_spectra's elements (matched by (y, z) in wavelengths)."""
    from rf_spectra import _element_offsets
    y, z = _element_offsets(M)
    pos = ant_pos.cpu().numpy() / lam                                    # [M^2, 3], x = 0
    perm = []
    for yi, zi in zip(y.numpy(), z.numpy()):
        d = (pos[:, 1] - yi) ** 2 + (pos[:, 2] - zi) ** 2
        j = int(np.argmin(d))
        if d[j] > 1e-6:
            raise SystemExit(f"no PanelArray element at ({yi}, {zi}) wavelengths")
        perm.append(j)
    if len(set(perm)) != M * M:
        raise SystemExit("element mapping is not a permutation")
    return torch.as_tensor(perm)


def spectrum(a, tau_ns, grid, loading, dt):
    """a [M^2, P] complex, tau_ns [P] -> native-dB MVDR spectrum [H, W] (the truth's processing)."""
    from rf_spectra import merge_paths_to_time_grid, mvdr_spectrum
    keep = torch.isfinite(tau_ns) & (tau_ns >= 0) & (a.abs().sum(0) > 0)
    a, tau_ns = a[:, keep], tau_ns[keep]
    tg = torch.arange(0, max(math.ceil(float(tau_ns.max())), 1), dt, device=a.device, dtype=tau_ns.dtype)
    resp = merge_paths_to_time_grid(a, tau_ns, tg)
    return mvdr_spectrum(resp, grid, diagonal_loading=loading)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out-prefix", default=os.path.join(REPO, "output", "rrf", "t4_"))
    ap.add_argument("--which", default="control,los,nn,inh_open,inh_mixed")
    ap.add_argument("--realisations", type=int, default=10)
    ap.add_argument("--loading", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0", help="Sionna accepts cuda:<i>, not cuda")
    ap.add_argument("--floor-offset", type=float, default=1.713, help="m added to every z for InH (scene z = 0 above the floor)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: n held-out positions spread along the route (end points included) with all their "
                         "faces, plus the position whose strongest face is weakest")
    a = ap.parse_args()
    dev = torch.device(a.device)
    meta = json.load(open(os.path.join(a.truth, "generation_meta.json")))
    M, fc, dt = meta["M"], meta["frequency"], meta["time_interval_ns"]
    lam = C0 / fc
    tx = np.array(meta["tx_loc"], dtype=np.float64)
    views, train, test = read_views(a.truth)
    if a.limit:
        # stratified, not a prefix: whole positions (every face = every orientation), spread over the route
        pos = {}
        for n in test:
            pos.setdefault(tuple(np.round(views[n][0], 6)), []).append(n)
        keys = sorted(pos, key=lambda k: min(pos[k]))                     # the route's order = the index's
        pick = {keys[j] for j in np.linspace(0, len(keys) - 1, a.limit).round().astype(int)}
        peak = {k: max(float(np.load(os.path.join(a.truth, "spectra_float", n + ".npy")).max()) for n in pos[k]) for k in keys}
        pick.add(min(keys, key=lambda k: peak[k]))
        test = [n for n in test if tuple(np.round(views[n][0], 6)) in pick]
        print(f"smoke: {len(pick)} positions, {len(test)} views (weakest position max {min(peak.values()):.1f} dB)")
    from generate_dataset import element_gain_fn
    from rf_spectra import ArrayGrid, steering_vector
    from t6_incoherent_mvdr import channel_stats
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono_polarized" if a.device.startswith("cuda") else "llvm_ad_mono_polarized")
    grid = ArrayGrid.build(M, meta["width"], meta["height"], meta["fov_deg"], element_gain_fn=element_gain_fn, device=dev)
    which = a.which.split(",")
    log = {"truth": a.truth, "loading": a.loading, "views": len(test)}

    def local_angles(d_world, yaw):
        """Zenith / azimuth of a world direction in the array frame of a face with this yaw (boresight = +x)."""
        x, y, z = d_world
        c, s = math.cos(yaw), math.sin(yaw)
        xl, yl = c * x + s * y, -s * x + c * y
        return math.acos(max(-1.0, min(1.0, z))), math.atan2(yl, xl)

    def element_gain(theta, phi):
        t = torch.tensor([[theta]], device=dev, dtype=torch.float32); p = torch.tensor([[phi]], device=dev, dtype=torch.float32)
        return element_gain_fn(t, p)[0, 0]

    def write(run, name, db):
        os.makedirs(os.path.join(run, "renders"), exist_ok=True)
        np.save(os.path.join(run, "renders", name + ".npy"), db.float().cpu().numpy())

    # ---- control: one ray, known direction, through PanelArray-style response and rf_spectra's manifold --------
    if "control" in which:
        from sionna.phy.channel.tr38901 import PanelArray
        arr = PanelArray(num_rows_per_panel=M, num_cols_per_panel=M, polarization="single", polarization_type="V",
                         antenna_pattern="38.901", carrier_frequency=fc, device=str(dev))
        perm = permutation(M, lam, arr.ant_pos)
        # directions = the centres of random pixels at least 10 px inside the image, the same for both paths. (The
        # first version drew angles up to +-40 deg azimuth / +-30 deg elevation, some outside the 90 x 67.4 deg face,
        # whose peaks then sat on the border: 2.22 deg max through PanelArray vs 0.19 through rf_spectra from two
        # different draws -- a flaw of the control, recorded in the notes)
        H_, W_ = grid.theta.shape[-2:]
        rng = np.random.default_rng(0)
        pix = [(int(rng.integers(10, H_ - 10)), int(rng.integers(10, W_ - 10))) for _ in range(40)]

        def peak_err(a_el, th, ph):
            db = spectrum(a_el, torch.tensor([5.0], device=dev), grid, 1e-3, dt)   # one ray: rank one, loading
            i = int(torch.argmax(db)); hh, ww = divmod(i, db.shape[1])
            t_hat, p_hat = float(grid.theta[hh, ww]), float(grid.phi[hh, ww])
            u = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            v = np.array([math.sin(t_hat) * math.cos(p_hat), math.sin(t_hat) * math.sin(p_hat), math.cos(t_hat)])
            return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v)))))

        errs, errs_ref = [], []
        for hh, ww in pix:
            th, ph = float(grid.theta[hh, ww]), float(grid.phi[hh, ww])
            # the plane-wave phase over the PanelArray's own element positions, then permuted
            k = torch.tensor([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)], device=dev)
            ph_el = 2 * math.pi / lam * (arr.ant_pos.to(dev).float() @ k)
            errs.append(peak_err(torch.exp(1j * ph_el.to(torch.complex64))[perm][:, None] * element_gain(th, ph), th, ph))
            # rf_spectra's own steering for the same direction, the reference convention
            a_ref = steering_vector(M, torch.tensor(th, device=dev), torch.tensor(ph, device=dev))[:, None] * element_gain(th, ph)
            errs_ref.append(peak_err(a_ref, th, ph))
        log["control"] = {"directions": len(pix), "panelarray_max_err_deg": max(errs), "rf_spectra_max_err_deg": max(errs_ref),
                          "first_version_failed": "2.22 deg (directions outside the face)"}
        print(f"control: single ray at {len(pix)} pixel directions, peak error max {max(errs):.2f} deg through PanelArray "
              f"positions, {max(errs_ref):.2f} deg through rf_spectra's steering (pixel pitch ~0.3 deg)")

    # ---- LOS: one free-space direct path ----------------------------------------------------------------------
    if "los" in which:
        run = a.out_prefix + "los"
        for n in test:
            rx, yaw = views[n]
            d = tx - rx; dist = float(np.linalg.norm(d))
            th, ph = local_angles(d / dist, yaw)                                    # at the Rx: towards the Tx
            th_t, ph_t = local_angles(-d / dist, 0.0)                               # at the Tx: towards the Rx
            amp = lam / (4 * math.pi * dist) * element_gain(th_t, ph_t) * element_gain(th, ph)
            a_el = steering_vector(M, torch.tensor(th, device=dev), torch.tensor(ph, device=dev))[:, None] * amp
            write(run, n, spectrum(a_el, torch.tensor([dist / C0 * 1e9], device=dev), grid, a.loading, dt))
        print(f"los: {len(test)} views -> {run}")

    # ---- NN: the nearest training position's truth, same face ---------------------------------------------------
    if "nn" in which:
        run = a.out_prefix + "nn"
        tr_pos = {}
        for n in train:
            rx, yaw = views[n]
            tr_pos.setdefault(round(yaw, 3), []).append((n, rx))
        for n in test:
            rx, yaw = views[n]
            cand = tr_pos[round(yaw, 3)]
            best = min(cand, key=lambda c: float(np.linalg.norm(c[1] - rx)))[0]
            os.makedirs(os.path.join(run, "renders"), exist_ok=True)
            np.save(os.path.join(run, "renders", n + ".npy"),
                    np.load(os.path.join(a.truth, "spectra_float", best + ".npy")).astype(np.float32))
        print(f"nn: {len(test)} views -> {run}")

    # ---- 3GPP InH ------------------------------------------------------------------------------------------------
    for kind in ("open", "mixed"):
        if f"inh_{kind}" not in which:
            continue
        import sionna
        from sionna.phy.channel.tr38901 import InH, PanelArray
        ut = PanelArray(num_rows_per_panel=M, num_cols_per_panel=M, polarization="single", polarization_type="V",
                        antenna_pattern="38.901", carrier_frequency=fc, device=str(dev))
        bs = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization="single", polarization_type="V",
                        antenna_pattern="38.901", carrier_frequency=fc, device=str(dev))
        perm = permutation(M, lam, ut.ant_pos).to(dev)
        # heights above the floor for 38.901: the scene's z = 0 is 1.713 m above the lobby floor (the NIST route
        # puts the receivers at 1.625 - 1.713 + U(-1, 0.3), generate_dataset.load_rx_locations); a common vertical
        # shift leaves every distance and angle unchanged and gives the BS 2.0 m, the UTs 0.6-1.9 m
        up = np.array([0.0, 0.0, a.floor_offset])
        rx_all = torch.tensor(np.stack([views[n][0] + up for n in test]), dtype=torch.float32, device=dev)[None]
        yaw_all = torch.tensor([[views[n][1], 0.0, 0.0] for n in test], dtype=torch.float32, device=dev)[None]
        for r in range(a.realisations):
            sionna.phy.config.seed = 1000 * (1 if kind == "open" else 2) + r
            model = InH(fc, ut, bs, "downlink", indoor_scenario=kind, enable_pathloss=True, enable_shadow_fading=True,
                        device=str(dev), enable_spatial_consistency=True)
            model.set_topology(ut_loc=rx_all, bs_loc=torch.tensor(tx + up, dtype=torch.float32, device=dev)[None, None],
                               ut_orientations=yaw_all, bs_orientations=torch.zeros(1, 1, 3, device=dev),
                               ut_velocities=torch.zeros_like(rx_all), in_state=torch.ones(1, len(test), dtype=torch.bool, device=dev))   # InH: every UT indoor
            t0 = time.time()
            h, tau = model(1, 1.0)
            if r == 0:
                # spatial consistency: the faces of one position are co-located UTs and must see one channel
                # (the same cluster delays and powers; only the array orientation differs)
                by_pos = {}
                for i, n in enumerate(test):
                    by_pos.setdefault(tuple(np.round(views[n][0], 6)), []).append(i)
                pw = (h[0, :, :, 0, 0, :, 0].abs() ** 2).mean(1)                       # [UT, P]
                # live clusters only, sorted: Sionna pads each UT to a fixed cluster count with zero-power slots whose
                # delays are arbitrary (up to seconds), which an index-wise comparison mistook for 477 ns differences
                live = lambda i: torch.sort(tau[0, i, 0, pw[i] > 0]).values                    # noqa: E731
                dmax = max((float((live(i) - live(ix[0])).abs().max()) if live(i).numel() == live(ix[0]).numel()
                            else float("inf")) for ix in by_pos.values() for i in ix)
                pmax = max(float((pw[ix] - pw[ix[0]]).abs().max() / pw[ix[0]].max()) for ix in by_pos.values())
                log[f"inh_{kind}_face_consistency"] = {"max_delay_diff_s": dmax, "max_rel_cluster_power_diff": pmax}
                print(f"inh_{kind}: faces of one position -- max cluster-delay difference {dmax:.2e} s, max relative "
                      f"cluster-power difference {pmax:.2e} (element pattern differs with orientation)")
            run = a.out_prefix + f"inh_{kind}_r{r}"
            stats = {}
            for i, n in enumerate(test):
                dist = float(np.linalg.norm(tx - views[n][0]))
                a_el = h[0, i, :, 0, 0, :, 0][perm]                                       # [M^2, P]
                tau_ns = tau[0, i, 0, :].to(torch.float64) * 1e9 + dist / C0 * 1e9
                write(run, n, spectrum(a_el, tau_ns.float(), grid, a.loading, dt))
                live = a_el.abs().sum(0) > 0
                stats[n] = channel_stats(a_el[:, live], tau_ns[live], dt)             # T4 level 2, t6's estimators
            json.dump(stats, open(os.path.join(run, "stats.json"), "w"), indent=1)
            print(f"inh_{kind} r{r}: {len(test)} views in {time.time() - t0:.0f} s -> {run}")
    os.makedirs(os.path.dirname(a.out_prefix), exist_ok=True)
    json.dump(log, open(a.out_prefix + "log.json", "w"), indent=1)


if __name__ == "__main__":
    main()

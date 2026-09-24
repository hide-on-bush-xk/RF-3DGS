"""A local browser for the trained models: predicted spectrum against ground truth.

The debugging question this answers is "what does model X get wrong, and where":
pick one of the trained models, walk its held-out views, and see the prediction,
the target and their difference side by side with that view's own metrics and
the receiver's position on the floor plan.

A 3D tab loads the released model's .ply. Its camera stands at the current
held-out view's receiver pose with the dataset's own camera, so there the render
is that view's prediction (shown beside it); the arrow keys step views as on the
compare tab and move the camera with them, and walking or turning away from a
pose leaves the ground truth behind -- it then shows what the field looks like,
not whether it is right.

A runs tab queries index_db.py's SQLite index over everything under output/rrf
-- 192 runs at the time of writing, past what grep answers comfortably. The
connection is opened read-only, so whatever query the page sends cannot modify
the index; and the index is derived from the JSONs anyway, so the worst case is
rebuilding it.

Why a server rather than a file:// page: the error map reads pixels back out of
a canvas, which a file:// image taints, and the .ply files are 240 MB each and
have to stream.

Standard library only -- no torch, no Sionna -- so it runs in any environment.

    python rrf_gsplat/viewer.py [--port 8770]
    then open http://localhost:8770
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RELEASED = os.path.join(REPO, "RF-3DGS_dataset", "RF-3DGS_trained_RRF")
SOURCES = os.path.join(REPO, "RF-3DGS_dataset", "training-rf-spectrum")
OURS = os.path.join(REPO, "output", "rrf")
VISUAL = os.path.join(REPO, "output", "visual_at_rf")

# matplotlib's jet at the 256 levels the datasets were quantised to. The page
# builds a reverse lookup from this so it can turn a rendered colour back into a
# value and show a signed error map. Kept here rather than imported from jet.py
# so this file needs no torch.
_JET_SEG = {
    "r": [(0.00, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.00, 0.5)],
    "g": [(0.00, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0), (0.91, 0.0), (1.00, 0.0)],
    "b": [(0.00, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.00, 0.0)],
}


def _interp(xs_ys, x):
    """Piecewise-linear interpolation of one jet channel."""
    for (x0, y0), (x1, y1) in zip(xs_ys[:-1], xs_ys[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return xs_ys[-1][1]


def jet_lut(levels=256):
    """[levels][3] of 0-255 ints: the exact colours the PNGs were written with."""
    out = []
    for i in range(levels):
        t = i / (levels - 1)
        out.append([int(round(255 * _interp(_JET_SEG[c], t))) for c in "rgb"])
    return out


# --------------------------------------------------------------------------
# poses
# --------------------------------------------------------------------------
def _qvec2rotmat(w, x, y, z):
    """COLMAP quaternion -> 3x3, as nested lists (no numpy dependency)."""
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def read_poses(images_txt):
    """{image name without extension: (rx [3], boresight azimuth in degrees, forward [3], down [3])}.

    COLMAP stores the world-to-camera rotation R and t = -R(rx), so rx = -R^T t.
    The receiver's look direction is the camera's +z axis in world coordinates,
    i.e. the third column of R^T; its azimuth is what the floor plan draws.
    Forward and down (the camera's +z and +y in world coordinates) are what the
    3D tab needs to put its camera exactly where the held-out view was taken.
    """
    poses = {}
    with open(images_txt) as fid:
        for line in fid:
            p = line.split()
            if len(p) < 10 or not p[9].lower().endswith(".png"):
                continue
            R = _qvec2rotmat(*(float(v) for v in p[1:5]))
            t = [float(v) for v in p[5:8]]
            # rx = -R^T t
            rx = [-sum(R[k][i] * t[k] for k in range(3)) for i in range(3)]
            # look direction = R^T [0,0,1] = third row of R read down the columns
            d = [R[2][i] for i in range(3)]
            down = [R[1][i] for i in range(3)]
            poses[os.path.splitext(p[9])[0]] = (rx, math.degrees(math.atan2(d[1], d[0])), d, down)
    return poses


# --------------------------------------------------------------------------
# model discovery
# --------------------------------------------------------------------------
def _ply_for(model_dir):
    """The highest-iteration point_cloud.ply under a model directory, or None.

    Compared numerically: sorted as strings, iteration_7000 beats iteration_30000.
    """
    pc = os.path.join(model_dir, "point_cloud")
    if not os.path.isdir(pc):
        return None
    best = None
    for name in os.listdir(pc):
        m = re.match(r"iteration_(\d+)$", name)
        p = os.path.join(pc, name, "point_cloud.ply")
        if m and os.path.isfile(p) and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best[1] if best else None


def _png_size(path):
    """(width, height) from a PNG header, or None.

    IHDR is always the first chunk: 8 bytes of signature, a 4-byte length, the
    tag, then width and height as big-endian uint32. Read directly so this file
    keeps its no-Pillow, standard-library-only property.
    """
    try:
        with open(path, "rb") as fid:
            head = fid.read(24)
        sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        if head[:8] != sig or head[12:16] != b"IHDR":
            return None
        return struct.unpack(">II", head[16:24])
    except Exception:
        return None


def _newest_test(model_dir):
    """The newest <model>/test/ours_<it> that actually holds renders, or None."""
    tdir = os.path.join(model_dir, "test")
    if not os.path.isdir(tdir):
        return None
    cands = [d for d in os.listdir(tdir)
             if re.match(r"ours_\d+$", d) and os.path.isdir(os.path.join(tdir, d, "renders"))]
    if not cands:
        return None
    return sorted(cands, key=lambda d: int(d.split("_")[-1]))[-1]


def _naming(renders, names):
    """How a renders/ directory names its files: 'name' or 'index', else None.

    The released models number their renders by position in the camera list,
    which Scene sorts by image name, so file i is the i-th held-out name. Our
    own runs keep the original name instead. Both index the same view list, and
    telling them apart is the whole of what makes the two comparable.
    """
    got = {os.path.splitext(f)[0] for f in renders}
    if got == set(names):
        return "name"
    if got == {f"{i:05d}" for i in range(len(names))}:
        return "index"
    return None


def _per_view(model_dir, test, key="PSNR"):
    """{filename: value} from per_view.json for the given ours_<it>, or {}."""
    p = os.path.join(model_dir, "per_view.json")
    if not os.path.isfile(p):
        return {}
    try:
        d = json.load(open(p))
    except Exception:
        return {}
    block = d.get(test) or next(iter(d.values()), {})
    return block.get(key, {}) if isinstance(block, dict) else {}


def _agg(model_dir, test):
    """Aggregate PSNR/SSIM/LPIPS for a model, from whichever file carries them.

    The released models put them in results.json under the ours_<it> key. Our
    runs' results.json is a different shape entirely (config, db_range, ...),
    and the comparable numbers live in inria_metrics.json instead -- computed by
    the upstream metrics.py so that they mean the same thing as the released
    ones.
    """
    p = os.path.join(model_dir, "results.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p))
            block = d.get(test)
            if isinstance(block, dict) and "PSNR" in block:
                return {k: block[k] for k in ("PSNR", "SSIM", "LPIPS") if k in block}
        except Exception:
            pass
    p = os.path.join(model_dir, "inria_metrics.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p))
            if "PSNR" in d:
                return {k: d[k] for k in ("PSNR", "SSIM", "LPIPS") if k in d}
        except Exception:
            pass
    return {}


def _our_runs():
    """{spectrum: [(run name, run dir)]} for our runs trained on a released set.

    A run qualifies only if results.json says its source is one of the six
    released datasets; runs on regenerated or ablation data render different
    targets and cannot share a panel with the released models.
    """
    out = {}
    if not os.path.isdir(OURS):
        return out
    for name in sorted(os.listdir(OURS)):
        rdir = os.path.join(OURS, name)
        rp = os.path.join(rdir, "results.json")
        if not os.path.isfile(rp):
            continue
        try:
            src = (json.load(open(rp)).get("config") or {}).get("source", "")
        except Exception:
            continue
        m = re.search(r"training-rf-spectrum[/\\]3dgs_(.+?)_100", str(src).replace("\\", "/"))
        if m:
            out.setdefault(m.group(1), []).append((name, rdir))
    return out


def discover():
    """The six spectra, each with every prediction that can be shown against it.

    A "spectrum" is the unit here rather than a model, because the released
    RF-3DGS model and our own runs predict the *same* held-out views of the
    *same* dataset -- identical poses, identical split -- and the useful view is
    all of them beside one target, not each on its own.

    The target comes from the source dataset's images/, not from any model's
    gt/, so it is the same bytes for every prediction and exists even where no
    model does (TCBF has a dataset and our runs but no released model).

    Per-view metrics are resolved into arrays indexed by view here, so the page
    never has to know that the two layouts name their files differently.
    """
    spectra = []
    ours = _our_runs()
    for src_name in sorted(os.listdir(SOURCES)) if os.path.isdir(SOURCES) else []:
        m = re.match(r"3dgs_(.+)_100$", src_name)
        if not m:
            continue
        spectrum = m.group(1)
        src = os.path.join(SOURCES, src_name)
        idx_path = os.path.join(src, "test_index.txt")
        if not os.path.isfile(idx_path):
            continue
        names = sorted(l.strip() for l in open(idx_path) if l.strip())
        if not names:
            continue

        # The datasets do not share an image size -- cameras.txt says 300x200
        # everywhere, but AoD and Delay are 231x154 on disk -- and the visual
        # renders are kept one directory per size, so the room shown beside a
        # spectrum is sampled on the same grid as the spectrum.
        size = _png_size(os.path.join(src, "images", f"{names[0]}.png"))
        vdir = os.path.join(VISUAL, "%dx%d" % size) if size else None

        poses = {}
        imgs_txt = os.path.join(src, "sparse", "0", "images.txt")
        if os.path.isfile(imgs_txt):
            poses = read_poses(imgs_txt)

        # every candidate prediction: the released model first, then our runs
        cands = []
        rel_dir = os.path.join(RELEASED, src_name)
        if os.path.isdir(rel_dir):
            cands.append(("released", "RF-3DGS (released)", rel_dir, "released"))
        for run_name, run_dir in ours.get(spectrum, []):
            # drop the spectrum TOKEN, not a prefix of that length: sota_Delay_db
            # sliced by len("Delay")+1 would read "elay_db"
            label = "_".join(t for t in run_name.split("_") if t != spectrum) or run_name
            cands.append((run_name, label, run_dir, "ours"))

        preds = []
        for pid, label, mdir, kind in cands:
            test = _newest_test(mdir)
            if not test:
                continue
            rdir = os.path.join(mdir, "test", test, "renders")
            files = sorted(f for f in os.listdir(rdir) if f.endswith(".png"))
            naming = _naming(files, names)
            if naming is None:
                # a partial render, or a split this spectrum does not share;
                # skipped rather than shown against the wrong target
                continue
            pv = _per_view(mdir, test)
            keys = [f"{n}.png" for n in names] if naming == "name" \
                else [f"{i:05d}.png" for i in range(len(names))]
            preds.append({
                "id": pid, "label": label, "kind": kind, "test": test,
                "naming": naming,
                "dir": os.path.relpath(rdir, REPO).replace(os.sep, "/"),
                "metrics": _agg(mdir, test),
                "psnr": [pv.get(k) for k in keys],
                "ply": _ply_for(mdir) is not None,
                # when the run finished, so the page can open on the newest run as well as the best one
                "mtime": os.path.getmtime(os.path.join(mdir, "results.json"))
                         if kind == "ours" and os.path.isfile(os.path.join(mdir, "results.json")) else None,
            })
        if not preds:
            continue

        items = []
        for i, n in enumerate(names):
            rx, yaw, fwd, down = poses.get(n, (None, None, None, None))
            items.append({
                "i": i, "name": n,
                "rx": None if rx is None else [round(rx[0], 3), round(rx[1], 3), round(rx[2], 3)],
                "yaw": None if yaw is None else round(yaw, 1),
                # the camera's +z and +y in world coordinates, for the 3D tab
                "fwd": None if fwd is None else [round(v, 5) for v in fwd],
                "down": None if down is None else [round(v, 5) for v in down],
            })
        spectra.append({
            "name": spectrum,
            "src": os.path.relpath(src, REPO).replace(os.sep, "/"),
            "views": len(names), "items": items, "preds": preds,
            "size": list(size) if size else None,
            "visual_dir": None if not vdir else os.path.relpath(vdir, REPO).replace(os.sep, "/"),
            "visual": bool(vdir and os.path.isfile(os.path.join(vdir, f"{names[0]}.png"))),
            "released": any(p["kind"] == "released" for p in preds),
        })
    spectra.extend(_regen_groups())
    return spectra


def _regen_groups():
    """The curated regenerated-data groups written by build_wall.py (output/rrf/wall_groups.json): our fields next to
    T4's predictors (NN, LOS, 3GPP InH, the float64 truth) on one regenerated dataset's held-out views. Each prediction
    carries per-view PSNR(jet) and main-peak direction error (deg); "baseline" marks the non-learned predictors."""
    p = os.path.join(OURS, "wall_groups.json")
    if not os.path.isfile(p):
        return []
    out = []
    for g in json.load(open(p)):
        src = os.path.join(REPO, g["dataset"])
        names = sorted(l.strip() for l in open(os.path.join(src, "test_index.txt")) if l.strip())
        poses = read_poses(os.path.join(src, "sparse", "0", "images.txt"))
        preds = []
        for e in g["preds"]:
            w = json.load(open(os.path.join(REPO, e["wall"])))
            run_res = os.path.join(OURS, e["id"], "results.json")
            preds.append({
                "id": e["id"], "label": e["label"], "kind": e["kind"], "default": e["default"],
                "test": f"ours_{w.get('iterations') or 0}", "naming": "name", "dir": e["dir"],
                "metrics": {"PSNR": w["PSNR"], "peak_median": w["peak_median"], "peak_within_1deg": w["peak_within_1deg"]},
                "psnr": [w["per_view_psnr"].get(n) for n in names],
                "peak": [w["peak_deg"].get(n) for n in names],
                "ply": False,
                "mtime": os.path.getmtime(run_res) if os.path.isfile(run_res) else None,
            })
        items = []
        for i, n in enumerate(names):
            rx, yaw, fwd, down = poses.get(n, (None, None, None, None))
            items.append({"i": i, "name": n,
                          "rx": None if rx is None else [round(rx[0], 3), round(rx[1], 3), round(rx[2], 3)],
                          "yaw": None if yaw is None else round(yaw, 1),
                          "fwd": None if fwd is None else [round(v, 5) for v in fwd],
                          "down": None if down is None else [round(v, 5) for v in down]})
        out.append({"name": g["name"], "src": g["dataset"], "views": len(names), "items": items, "preds": preds,
                    "size": list(_png_size(os.path.join(src, "images", f"{names[0]}.png")) or []) or None,
                    "visual_dir": None, "visual": False, "released": False, "regen": True})
    return out




# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------
DEFAULT_RUNS_SQL = """SELECT r.name, r.source, r.mode, r.geometry, r.opacity,
       r.iterations, r.n_train, r.gaussians,
       round(r.train_s) AS train_s, round(r.it_per_s, 1) AS it_per_s,
       round(MAX(CASE WHEN m.key='psnr_rgb'         THEN m.value END), 2) AS psnr,
       round(MAX(CASE WHEN m.key='ssim_rgb'         THEN m.value END), 3) AS ssim,
       round(MAX(CASE WHEN m.key='rmse_db'          THEN m.value END), 2) AS rmse_db,
       round(MAX(CASE WHEN m.key='rmse_db_in_range' THEN m.value END), 3) AS rmse_in_range
FROM runs r LEFT JOIN metrics m ON m.run = r.name
GROUP BY r.name ORDER BY r.mtime DESC"""


def make_handler(spectra, page, db_path=None):
    by_name = {s["name"]: s for s in spectra}

    # Resolve every servable image to an absolute path once, indexed by view, so
    # a request is a list lookup and the two file-naming conventions stop
    # mattering past this point.
    pred_files, gt_files, visual_files = {}, {}, {}
    for s in spectra:
        names = [it["name"] for it in s["items"]]
        gt_files[s["name"]] = [os.path.join(REPO, s["src"], "images", f"{n}.png") for n in names]
        vdir = os.path.join(REPO, s["visual_dir"]) if s["visual_dir"] else None
        visual_files[s["name"]] = None if not vdir else             [os.path.join(vdir, f"{n}.png") for n in names]
        for p in s["preds"]:
            base = os.path.join(REPO, p["dir"])
            files = [f"{n}.png" for n in names] if p["naming"] == "name" \
                else [f"{i:05d}.png" for i in range(len(names))]
            pred_files[(s["name"], p["id"])] = [os.path.join(base, f) for f in files]

    index = {"spectra": spectra, "jet": jet_lut(),
             "db": bool(db_path), "default_sql": DEFAULT_RUNS_SQL}
    index_bytes = json.dumps(index).encode()

    # One read-only connection for the process, shared across handler threads
    # under a lock -- sqlite3 allows the sharing with check_same_thread=False but
    # a cursor is not safe to interleave. Read-only is what makes an arbitrary
    # query from the page harmless, without having to parse SQL looking for
    # dangerous statements.
    con = None
    con_lock = threading.Lock()
    if db_path:
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                              check_same_thread=False)

    class H(BaseHTTPRequestHandler):
        # HTTP/1.1 is needed for Range, and it brings keep-alive with it: the
        # browser holds the connection open after a response. That MUST be paired
        # with a threading server. On a single-threaded one the first connection
        # is served until the client closes it and every other connection sits in
        # the accept backlog -- a browser opens up to six per origin, so the page
        # loaded once and then any request that happened to go out on one of the
        # idle connections hung forever. Images stopped arriving and every button
        # looked dead, because each click re-entered the same stalled fetch.
        protocol_version = "HTTP/1.1"

        def _send(self, body, ctype, code=200, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, ctype):
            """Serve a file, honouring a single Range request.

            The .ply files are 240 MB; without Range the browser cannot resume
            and some loaders refuse to start.
            """
            if not os.path.isfile(path):
                self._send(b"not found", "text/plain", 404); return
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            code = 200
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:                               # suffix range
                        start = max(0, size - int(m.group(2)))
                    end = min(end, size - 1)
                    code = 206
            with open(path, "rb") as fid:
                fid.seek(start)
                body = fid.read(end - start + 1)
            extra = {"Accept-Ranges": "bytes"}
            if code == 206:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self._send(body, ctype, code, extra)

        def do_GET(self):
            from urllib.parse import unquote
            # decoded, because the regenerated groups' names carry spaces and brackets and the browser sends them
            # percent-encoded; no name contains "/", so decoding cannot change how the path splits
            p = unquote(self.path.split("?")[0])
            if p in ("/", "/index.html"):
                self._send(page.encode(), "text/html; charset=utf-8"); return
            if p == "/api/index":
                self._send(index_bytes, "application/json"); return
            if p == "/api/sql":
                self._sql(); return
            # every image is addressed by VIEW INDEX, never by filename, so the
            # page can ask for "view 137 of MVDR" without knowing which of the
            # two naming conventions the prediction on disk happens to use
            m = re.match(r"^/gt/([^/]+)/(\d+)$", p)
            if m:
                self._by_index(gt_files.get(m.group(1)), m.group(2)); return
            m = re.match(r"^/visual/([^/]+)/(\d+)$", p)
            if m:
                self._by_index(visual_files.get(m.group(1)), m.group(2)); return
            m = re.match(r"^/pred/([^/]+)/([^/]+)/(\d+)$", p)
            if m:
                self._by_index(pred_files.get((m.group(1), m.group(2))), m.group(3)); return
            m = re.match(r"^/ply/([^/]+)$", p)
            if m and m.group(1) in by_name:
                mdir = os.path.join(RELEASED, f"3dgs_{m.group(1)}_100")
                ply = _ply_for(mdir)
                if ply:
                    self._file(ply, "application/octet-stream"); return
            self._send(b"not found", "text/plain", 404)

        def _by_index(self, files, raw):
            """Serve files[int(raw)], or 404 if the list or the index is absent."""
            if files is None:
                self._send(b"not found", "text/plain", 404); return
            i = int(raw)
            if not 0 <= i < len(files):
                self._send(b"view out of range", "text/plain", 404); return
            self._file(files[i], "image/png")

        def _sql(self):
            """Run one query against the read-only index and return columns + rows.

            A row cap is applied because the history table alone is 16k rows and
            an accidental `SELECT * FROM history` would otherwise try to render
            all of it. The error text is passed through verbatim: this is a
            local tool and SQLite's messages say exactly what is wrong.
            """
            from urllib.parse import parse_qs, urlparse
            if con is None:
                self._send(json.dumps({"error": "no index.db; run rrf_gsplat/index_db.py"}).encode(),
                           "application/json"); return
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip()
            limit = int(parse_qs(urlparse(self.path).query).get("limit", ["2000"])[0])
            if not q:
                self._send(json.dumps({"error": "empty query"}).encode(), "application/json"); return
            try:
                with con_lock:
                    cur = con.execute(q)
                    cols = [c[0] for c in (cur.description or [])]
                    rows = cur.fetchmany(limit)
                    more = cur.fetchone() is not None
                body = {"cols": cols, "rows": rows, "truncated": more, "limit": limit}
            except Exception as exc:
                body = {"error": f"{type(exc).__name__}: {exc}"}
            self._send(json.dumps(body, default=str).encode(), "application/json")

        def log_message(self, fmt, *args):
            pass                                        # silence the access log

    return H


def main():
    """Discover the models, then serve the page on localhost."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--print-index", action="store_true",
                    help="dump the discovered index and exit, for checking the mapping")
    cfg = ap.parse_args()
    spectra = discover()
    if not spectra:
        raise SystemExit(f"no rendered predictions found under {RELEASED} or {OURS}")
    for s in spectra:
        got = sum(1 for it in s["items"] if it["rx"] is not None)
        sz = "%dx%d" % tuple(s["size"]) if s["size"] else "size unknown"
        vis = f"visual {sz}" if s["visual"] else "NO VISUAL (MISSING)"
        print(f"  {s['name']:6s} {s['views']:4d} views, {got} with a pose, {sz}, {vis}")
        for p in s["preds"]:
            # MISSING rather than a blank: a prediction with no comparable
            # aggregate is still a row, it just has no number yet
            psnr = p["metrics"].get("PSNR")
            print(f"      {p['kind']:8s} {p['label']:16s} {p['test']:11s} "
                  f"by {p['naming']:5s}  PSNR "
                  f"{'MISSING' if psnr is None else format(psnr, '.2f')}")
    if cfg.print_index:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    page = open(os.path.join(here, "viewer.html"), encoding="utf-8").read()
    # The index is optional: without it the runs tab says so and the rest works.
    db = os.path.join(OURS, "index.db")
    if os.path.isfile(db):
        import sqlite3
        n = sqlite3.connect(f"file:{db}?mode=ro", uri=True) \
            .execute("SELECT count(*) FROM runs").fetchone()[0]
        print(f"  index.db: {n} runs")
    else:
        print(f"  no index.db (build it with {os.path.join('rrf_gsplat', 'index_db.py')})")
        db = None
    print(f"\nserving on http://localhost:{cfg.port}")
    # ThreadingHTTPServer, not HTTPServer: see protocol_version in the handler.
    ThreadingHTTPServer(("127.0.0.1", cfg.port),
                        make_handler(spectra, page, db)).serve_forever()


if __name__ == "__main__":
    main()

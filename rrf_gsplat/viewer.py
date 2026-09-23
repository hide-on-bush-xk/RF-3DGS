"""A local browser for the trained models: predicted spectrum against ground truth.

The debugging question this answers is "what does model X get wrong, and where":
pick one of the trained models, walk its held-out views, and see the prediction,
the target and their difference side by side with that view's own metrics and
the receiver's position on the floor plan.

A 3D tab loads the model's .ply for a free look at the geometry. It is the
secondary feature: a free viewpoint has no ground truth to compare against, so
it shows what the field looks like, not whether it is right.

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
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RELEASED = os.path.join(REPO, "RF-3DGS_dataset", "RF-3DGS_trained_RRF")
SOURCES = os.path.join(REPO, "RF-3DGS_dataset", "training-rf-spectrum")
OURS = os.path.join(REPO, "output", "rrf")

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
    """{image name without extension: (rx [3], boresight azimuth in degrees)}.

    COLMAP stores the world-to-camera rotation R and t = -R(rx), so rx = -R^T t.
    The receiver's look direction is the camera's +z axis in world coordinates,
    i.e. the third column of R^T; its azimuth is what the floor plan draws.
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
            poses[os.path.splitext(p[9])[0]] = (rx, math.degrees(math.atan2(d[1], d[0])))
    return poses


# --------------------------------------------------------------------------
# model discovery
# --------------------------------------------------------------------------
def _ply_for(model_dir):
    """The highest-iteration point_cloud.ply under a model directory, or None."""
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


def discover():
    """Every model that has a rendered held-out set, with its per-view metadata.

    Only the released layout (<model>/test/ours_<it>/{renders,gt}) is handled
    here; our own runs write a flat renders/ directory with no gt beside it, so
    they need the source dataset to supply the target and are left out until
    that pairing is needed.
    """
    models = []
    if not os.path.isdir(RELEASED):
        return models
    for name in sorted(os.listdir(RELEASED)):
        m = re.match(r"3dgs_(.+)_100$", name)
        if not m:
            continue
        spectrum = m.group(1)
        mdir = os.path.join(RELEASED, name)
        # the newest ours_<it> that actually holds both sides
        tests = [d for d in os.listdir(os.path.join(mdir, "test"))
                 if os.path.isdir(os.path.join(mdir, "test", d, "renders"))] \
            if os.path.isdir(os.path.join(mdir, "test")) else []
        if not tests:
            continue
        test = sorted(tests, key=lambda d: int(d.split("_")[-1]))[-1]
        tdir = os.path.join(mdir, "test", test)
        renders = sorted(f for f in os.listdir(os.path.join(tdir, "renders")) if f.endswith(".png"))
        if not renders:
            continue

        # render index -> original name: render.py numbers by position in the
        # camera list, and Scene sorts cameras by image name, so the i-th render
        # is the i-th name of the sorted held-out split.
        src = os.path.join(SOURCES, name)
        names, poses = [], {}
        idx_path = os.path.join(src, "test_index.txt")
        if os.path.isfile(idx_path):
            names = sorted(l.strip() for l in open(idx_path) if l.strip())
            imgs = os.path.join(src, "sparse", "0", "images.txt")
            if os.path.isfile(imgs):
                poses = read_poses(imgs)

        pv = {}
        pvp = os.path.join(mdir, "per_view.json")
        if os.path.isfile(pvp):
            d = json.load(open(pvp))
            pv = d.get(test, next(iter(d.values()), {}))
        agg = {}
        rp = os.path.join(mdir, "results.json")
        if os.path.isfile(rp):
            d = json.load(open(rp))
            agg = d.get(test, next(iter(d.values()), {}))

        items = []
        for i, f in enumerate(renders):
            orig = names[i] if i < len(names) else None
            rx, yaw = poses.get(orig, (None, None))
            items.append({
                "i": i, "file": f, "name": orig,
                "psnr": pv.get("PSNR", {}).get(f), "ssim": pv.get("SSIM", {}).get(f),
                "lpips": pv.get("LPIPS", {}).get(f),
                "rx": None if rx is None else [round(rx[0], 3), round(rx[1], 3), round(rx[2], 3)],
                "yaw": None if yaw is None else round(yaw, 1),
            })
        models.append({
            "name": spectrum, "dir": os.path.relpath(tdir, REPO).replace(os.sep, "/"),
            "test": test, "views": len(items), "metrics": agg,
            "ply": _ply_for(mdir) is not None, "items": items,
        })
    return models


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


def make_handler(models, page, db_path=None):
    by_name = {m["name"]: m for m in models}
    index = {"models": [{k: v for k, v in m.items()} for m in models], "jet": jet_lut(),
             "db": bool(db_path), "default_sql": DEFAULT_RUNS_SQL}
    index_bytes = json.dumps(index).encode()

    # One read-only connection for the process. HTTPServer here is
    # single-threaded, so a single connection is safe; read-only is what makes
    # an arbitrary query from the page harmless, without having to parse SQL
    # looking for dangerous statements.
    con = None
    if db_path:
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                              check_same_thread=False)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"          # needed for Range / keep-alive

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
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                self._send(page.encode(), "text/html; charset=utf-8"); return
            if p == "/api/index":
                self._send(index_bytes, "application/json"); return
            if p == "/api/sql":
                self._sql(); return
            # /img/<model>/<renders|gt>/<file>
            m = re.match(r"^/img/([^/]+)/(renders|gt)/([0-9]+\.png)$", p)
            if m and m.group(1) in by_name:
                self._file(os.path.join(REPO, by_name[m.group(1)]["dir"], m.group(2), m.group(3)),
                           "image/png")
                return
            m = re.match(r"^/ply/([^/]+)$", p)
            if m and m.group(1) in by_name:
                mdir = os.path.join(RELEASED, f"3dgs_{m.group(1)}_100")
                ply = _ply_for(mdir)
                if ply:
                    self._file(ply, "application/octet-stream"); return
            self._send(b"not found", "text/plain", 404)

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
    models = discover()
    if not models:
        raise SystemExit(f"no rendered models found under {RELEASED}")
    for m in models:
        got = sum(1 for it in m["items"] if it["rx"] is not None)
        print(f"  {m['name']:6s} {m['views']:4d} views, {got} with a pose, "
              f"PSNR {m['metrics'].get('PSNR', float('nan')):.2f}, ply {'yes' if m['ply'] else 'no'}")
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
    HTTPServer(("127.0.0.1", cfg.port), make_handler(models, page, db)).serve_forever()


if __name__ == "__main__":
    main()

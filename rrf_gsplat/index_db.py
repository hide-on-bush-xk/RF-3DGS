"""Index every run under output/rrf into one queryable SQLite file.

At 192 runs the results are past the point where grep and a text editor answer
questions like "every db-mode run with unfrozen geometry, ordered by in-range
RMSE" or "which runs never reached 15 dB". This scans what the runs already
wrote and builds an index over it.

The index is DERIVED, never authoritative. The JSON files under output/rrf are
what the runs produced and remain the only source of truth; this file can be
deleted and rebuilt in seconds, and nothing depends on it surviving. Blobs stay
on disk for the same reason and because a filesystem serves them better than a
database can -- 30 GB of .npy renders would lose HTTP Range, the OS page cache
and any sane backup. The artifacts table holds paths, never contents.

SQLite rather than a server database because it is in the standard library (so
this and viewer.py keep running in any environment), because a 1 MB index can
be handed to the browser and queried there with a WASM build, and because
"delete it and rebuild" is the right recovery story for derived data.

    python rrf_gsplat/index_db.py                 # build or refresh
    python rrf_gsplat/index_db.py --rebuild       # from scratch
    python rrf_gsplat/index_db.py --check         # verify against the JSONs
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RRF = os.path.join(REPO, "output", "rrf")
DB = os.path.join(RRF, "index.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  name       TEXT PRIMARY KEY,
  dir        TEXT NOT NULL,
  kind       TEXT NOT NULL,          -- trainer | metrics-only
  source     TEXT, mode TEXT, sh_degree INTEGER, iterations INTEGER,
  n_train    INTEGER, n_test INTEGER, gaussians INTEGER,
  opacity    TEXT, geometry TEXT, densify TEXT, warm TEXT,
  train_s    REAL, it_per_s REAL, total_s REAL,
  db_lo      REAL, db_hi REAL,
  channels   TEXT,                   -- JSON list, or NULL
  mtime      REAL
);
-- config and metrics are long, not wide, because they are sparse: 42 config
-- keys and 12 metric keys, and rmse_aod_az appears in only 23 % of runs. A wide
-- table would be mostly NULL and would need a migration for every new metric.
CREATE TABLE IF NOT EXISTS config (
  run TEXT NOT NULL, key TEXT NOT NULL,
  value TEXT,                        -- always set; JSON for non-scalars
  num   REAL,                        -- set when the value is numeric
  PRIMARY KEY (run, key)
);
CREATE TABLE IF NOT EXISTS metrics (
  run TEXT NOT NULL, key TEXT NOT NULL, value REAL,
  PRIMARY KEY (run, key)
);
CREATE TABLE IF NOT EXISTS history (
  run TEXT NOT NULL, iteration INTEGER, key TEXT NOT NULL, value REAL
);
-- Pointers to what is on disk. Never the bytes themselves.
CREATE TABLE IF NOT EXISTS artifacts (
  run TEXT NOT NULL, kind TEXT NOT NULL, path TEXT, n INTEGER, bytes INTEGER
);
CREATE INDEX IF NOT EXISTS ix_config_key  ON config(key, value);
CREATE INDEX IF NOT EXISTS ix_config_num  ON config(key, num);
CREATE INDEX IF NOT EXISTS ix_metrics_key ON metrics(key, value);
CREATE INDEX IF NOT EXISTS ix_history_key ON history(run, key, iteration);
CREATE INDEX IF NOT EXISTS ix_art_run     ON artifacts(run, kind);
"""


def _num(v):
    """The numeric value of a JSON scalar, or None. bool is deliberately
    excluded: True would index as 1 and silently match a numeric filter."""
    if isinstance(v, bool) or v is None:
        return None
    return float(v) if isinstance(v, (int, float)) else None


def _scalar_text(v):
    """A stable text form: scalars as themselves, everything else as JSON."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    return json.dumps(v, sort_keys=True)


def scan_run(d):
    """Everything one run directory contributes, or None if it holds no results.

    Two shapes are accepted. A trainer run has config/final/history. A run whose
    results.json was overwritten by metrics.py has only "ours_<iteration>" keys:
    inria_metrics.py restores the trainer's copy afterwards, but three runs in
    this tree were left in that state, so they are indexed as metrics-only
    rather than dropped without a word.
    """
    p = os.path.join(d, "results.json")
    if not os.path.isfile(p):
        return None
    try:
        r = json.load(open(p, encoding="utf-8"))
    except Exception as exc:
        return {"broken": f"{type(exc).__name__}: {exc}"}

    name = os.path.basename(d.rstrip(os.sep))
    rec = {"name": name, "dir": os.path.relpath(d, REPO).replace(os.sep, "/"),
           "mtime": os.path.getmtime(p), "config": {}, "metrics": {}, "history": [],
           "artifacts": []}

    if isinstance(r.get("config"), dict):
        rec["kind"] = "trainer"
        c = r["config"]
        rec["config"] = c
        final = r.get("final") or {}
        rec["metrics"] = {k: v for k, v in final.items() if _num(v) is not None}
        for e in (r.get("history") or []):
            it = e.get("iteration")
            for k, v in e.items():
                if k != "iteration" and _num(v) is not None:
                    rec["history"].append((it, k, float(v)))
        rng = r.get("db_range") or [None, None]
        rec["row"] = {
            "source": os.path.basename(str(c.get("source", "")).rstrip("/")),
            "mode": c.get("mode"), "sh_degree": c.get("sh_degree"),
            "iterations": c.get("iterations"),
            "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "gaussians": r.get("gaussians"),
            "opacity": "frozen" if c.get("freeze_opacity") else "trained",
            "geometry": "trained" if c.get("train_geometry") else "frozen",
            "densify": c.get("densify", "none"),
            "warm": os.path.basename(os.path.dirname(c["init_from"])) if c.get("init_from") else None,
            "train_s": r.get("train_seconds"), "it_per_s": r.get("iters_per_second"),
            "total_s": r.get("total_seconds"),
            "db_lo": rng[0] if isinstance(rng, list) and len(rng) == 2 else None,
            "db_hi": rng[1] if isinstance(rng, list) and len(rng) == 2 else None,
            "channels": json.dumps(r["channels"]) if r.get("channels") else None,
        }
    else:
        # metrics.py's own shape: {"ours_<it>": {PSNR, SSIM, LPIPS}}
        rec["kind"] = "metrics-only"
        rec["row"] = {k: None for k in
                      ("source", "mode", "sh_degree", "iterations", "n_train", "n_test",
                       "gaussians", "opacity", "geometry", "densify", "warm",
                       "train_s", "it_per_s", "total_s", "db_lo", "db_hi", "channels")}
        for key, block in r.items():
            if isinstance(block, dict):
                for k, v in block.items():
                    if _num(v) is not None:
                        rec["metrics"][f"{key}.{k}"] = float(v)

    # artifacts: the sibling JSONs, and the render directory as a count, never
    # its contents.
    for f in sorted(os.listdir(d)):
        fp = os.path.join(d, f)
        if f.endswith(".json") and os.path.isfile(fp):
            rec["artifacts"].append((f[:-5], os.path.relpath(fp, REPO).replace(os.sep, "/"),
                                     1, os.path.getsize(fp)))
    rd = os.path.join(d, "renders")
    if os.path.isdir(rd):
        n = b = 0
        with os.scandir(rd) as it:
            for e in it:
                if e.is_file():
                    n += 1; b += e.stat().st_size
        rec["artifacts"].append(("renders", os.path.relpath(rd, REPO).replace(os.sep, "/"), n, b))
    return rec


def build(db_path=DB, rebuild=False, verbose=True):
    """Scan every run directory and write the index. Returns a summary dict."""
    if rebuild and os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    dirs = sorted(d for d in (os.path.join(RRF, x) for x in os.listdir(RRF))
                  if os.path.isdir(d))
    n_ok = n_skip = 0
    broken = []
    t0 = time.time()
    for d in dirs:
        rec = scan_run(d)
        if rec is None:
            n_skip += 1
            continue
        if "broken" in rec:
            broken.append((os.path.basename(d), rec["broken"]))
            continue
        name = rec["name"]
        # Rewritten wholesale rather than merged: the JSONs are the truth, so a
        # rerun that dropped a metric must drop it here too.
        for t in ("config", "metrics", "history", "artifacts"):
            con.execute(f"DELETE FROM {t} WHERE run = ?", (name,))
        row = rec["row"]
        con.execute(
            "INSERT OR REPLACE INTO runs (name,dir,kind,source,mode,sh_degree,iterations,"
            "n_train,n_test,gaussians,opacity,geometry,densify,warm,train_s,it_per_s,"
            "total_s,db_lo,db_hi,channels,mtime) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, rec["dir"], rec["kind"], row["source"], row["mode"], row["sh_degree"],
             row["iterations"], row["n_train"], row["n_test"], row["gaussians"],
             row["opacity"], row["geometry"], row["densify"], row["warm"],
             row["train_s"], row["it_per_s"], row["total_s"], row["db_lo"], row["db_hi"],
             row["channels"], rec["mtime"]))
        con.executemany("INSERT OR REPLACE INTO config VALUES (?,?,?,?)",
                        [(name, k, _scalar_text(v), _num(v)) for k, v in rec["config"].items()])
        con.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?)",
                        [(name, k, v) for k, v in rec["metrics"].items()])
        con.executemany("INSERT INTO history VALUES (?,?,?,?)",
                        [(name, it, k, v) for it, k, v in rec["history"]])
        con.executemany("INSERT INTO artifacts VALUES (?,?,?,?,?)",
                        [(name, k, p, n, b) for k, p, n, b in rec["artifacts"]])
        n_ok += 1
    con.commit()
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("runs", "config", "metrics", "history", "artifacts")}
    con.execute("VACUUM")
    con.close()
    size = os.path.getsize(db_path)
    if verbose:
        print(f"{n_ok} runs indexed, {n_skip} directories without results.json, "
              f"{len(broken)} unreadable, {time.time() - t0:.1f} s")
        for nm, why in broken:
            print(f"  unreadable: {nm}: {why}")
        print("  " + "  ".join(f"{t} {c:,}" for t, c in counts.items()))
        print(f"  {db_path} -> {size / 1048576:.2f} MB")
    return {"runs": n_ok, "skipped": n_skip, "broken": broken, "counts": counts, "bytes": size}


def connect_ro(db_path=DB):
    """A read-only connection.

    Anything the page can reach goes through this, so an arbitrary query cannot
    modify the index however it is phrased -- cheaper and more reliable than
    trying to parse SQL for dangerous statements.
    """
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def check(db_path=DB):
    """Re-read the JSONs and compare them with what the index holds."""
    con = connect_ro(db_path)
    bad = 0
    rows = con.execute("SELECT name, dir, kind FROM runs").fetchall()
    for name, d, kind in rows:
        p = os.path.join(REPO, d, "results.json")
        r = json.load(open(p, encoding="utf-8"))
        want = {k: float(v) for k, v in ((r.get("final") or {}) if kind == "trainer" else {}).items()
                if _num(v) is not None}
        got = dict(con.execute("SELECT key, value FROM metrics WHERE run = ?", (name,)).fetchall())
        if kind == "trainer" and want != {k: v for k, v in got.items()}:
            miss = set(want) ^ set(got)
            diff = [k for k in set(want) & set(got) if want[k] != got[k]]
            print(f"  MISMATCH {name}: missing/extra {sorted(miss)}, differing {diff}")
            bad += 1
    print(f"checked {len(rows)} runs against their JSON, {bad} mismatched")
    con.close()
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--rebuild", action="store_true", help="delete and recreate")
    ap.add_argument("--check", action="store_true", help="verify against the JSONs and exit")
    ap.add_argument("--sql", default=None, help="run one read-only query and print it")
    cfg = ap.parse_args()
    if cfg.check:
        raise SystemExit(1 if check(cfg.db) else 0)
    if cfg.sql:
        con = connect_ro(cfg.db)
        cur = con.execute(cfg.sql)
        cols = [c[0] for c in cur.description]
        print(" | ".join(cols))
        for row in cur.fetchall():
            print(" | ".join("" if v is None else str(v) for v in row))
        con.close()
        return
    build(cfg.db, cfg.rebuild)


if __name__ == "__main__":
    main()

"""Levenberg-Marquardt after Adam (3DGS-LM; Hoellein et al., ICCV 2025) for train_rrf.py's db mode.

3DGS-LM replaces the last third of 3DGS's Adam iterations by a few LM iterations. Each LM iteration solves the
normal equations (J^T J + C / radius) x = J^T r with preconditioned conjugate gradients (PCG) on a few strided
subsets of the images (C = diag(J^T J) clamped to [1, 1e6], preconditioner 1 / (diag + C)), averages the subsets'
solutions weighted by diag(J^T J), scales the step by a backtracking line search on 30 % of the images, and keeps
or undoes it by a Ceres-style trust region on the actual vs the linearised cost change (lukasHoel/3DGS-LM,
train.py lm_step / linear_solve_pcg_fused; its defaults are used below). Its speed comes from custom CUDA kernels
that cache per-pixel gradients so that J p and J^T z are cheap.

Here, with the geometry and the opacity frozen, the rendered value is linear in the SH coefficients (gsplat SH,
then alpha blending), so the products are exact without a cache:
  J p     one render with the coefficients set to p, minus the render at zero coefficients (the constant 0.5)
  J^T z   one backward of the render (its graph is kept for a subset's whole solve)
The residuals are 3DGS-LM's square-root form of the training loss, summed over pixels:
  r1 = sqrt((1 - lambda) |f - y| + eps),   r2 = sqrt(lambda (1 - ssim_map(f, y)) + eps)
and their Jacobian w.r.t. the image goes through torch.func (jvp / vjp, the SSIM convolutions included).
Differences from 3DGS-LM, stated where results are reported: only the SH coefficients move (the opacity stays
where Adam left it); diag(J^T J) is approximated from the L1 residuals as sum_p d_p^2 w_pj Y_k^2 (w <= 1, so an
upper bound of that part; the SSIM coupling is left out) instead of computed exactly; no gradient cache.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6


def ssim_window(device, size=11, sigma=1.5):
    x = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] @ g[None, :]).float().to(device)[None, None]           # [1, 1, 11, 11]


def ssim_map(img, gt, win):
    """[B, 1, H, W] -> per-pixel SSIM, the statistics of utils.loss_utils._ssim (C1 = 0.01^2, C2 = 0.03^2)."""
    p = win.shape[-1] // 2
    mu1, mu2 = F.conv2d(img, win, padding=p), F.conv2d(gt, win, padding=p)
    s11 = F.conv2d(img * img, win, padding=p) - mu1 * mu1
    s22 = F.conv2d(gt * gt, win, padding=p) - mu2 * mu2
    s12 = F.conv2d(img * gt, win, padding=p) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (s11 + s22 + c2))


def residuals(f, y, win, lam):
    r1 = torch.sqrt((1 - lam) * (f - y).abs() + EPS)
    r2 = torch.sqrt(lam * (1 - ssim_map(f, y, win)).clamp_min(0) + EPS)
    return torch.cat([r1.reshape(-1), r2.reshape(-1)])


@contextmanager
def coefficients(model, sh0, shN):
    """Temporarily render with other SH coefficients (no autograd on them)."""
    a, b = model.params["sh0"].data, model.params["shN"].data
    model.params["sh0"].data, model.params["shN"].data = sh0, shN
    try:
        yield
    finally:
        model.params["sh0"].data, model.params["shN"].data = a, b


class Group:
    """One receiver position (its faces rendered together, as training does) and its linear-algebra pieces."""

    def __init__(self, model, views, span, targets, win, lam):
        self.model, self.views, self.span, self.y, self.win, self.lam = model, views, span, targets, win, lam
        vm, K, W, H = views
        with torch.no_grad(), coefficients(model, torch.zeros_like(model.params["sh0"]),
                                           torch.zeros_like(model.params["shN"])):
            self.b0 = model.render_batch(vm, K, W, H, span)                 # the render at zero coefficients
        self.out = None

    def render(self):
        vm, K, W, H = self.views
        return self.model.render_batch(vm, K, W, H, self.span)

    def linearise(self):
        """Render at the current coefficients, keep the graph (for J^T z) and the residual Jacobian."""
        self.out = self.render()
        f = self.out.detach()
        fn = lambda img: residuals(img, self.y, self.win, self.lam)                     # noqa: E731
        self.r, self.vjp = torch.func.vjp(fn, f)
        self.fn, self.f = fn, f
        return self.r

    def Jp(self, p0, pN):
        with torch.no_grad(), coefficients(self.model, p0, pN):
            return self.render() - self.b0

    def JTz(self, z):
        return torch.autograd.grad(self.out, [self.model.params["sh0"], self.model.params["shN"]], grad_outputs=z,
                                   retain_graph=True)

    def res_jvp(self, u):
        return torch.func.jvp(self.fn, (self.f,), (u,))[1]

    def gradient(self):
        return self.JTz(self.vjp(self.r)[0])

    def jtj(self, p0, pN):
        return self.JTz(self.vjp(self.res_jvp(self.Jp(p0, pN)))[0])

    def diag(self, sh_basis, deg):
        """Approximate diag(J^T J) for the SH coefficients from the L1 residuals (see the module docstring)."""
        model = self.model
        vm = self.views[0]
        centre = torch.linalg.inv(vm)[0, :3, 3]
        col = model.colours(centre).detach().requires_grad_(True)
        model.colours = lambda c: col            # an instance attribute shadowing the method, removed below
        try:
            out = self.render()
        finally:
            del model.colours
        r1 = torch.sqrt((1 - self.lam) * (self.f - self.y).abs() + EPS)
        d2 = ((1 - self.lam) ** 2) / (4 * r1 * r1)
        u = torch.autograd.grad(out, col, grad_outputs=d2)[0]                          # [N, C]
        with torch.no_grad():
            Y = sh_basis(deg, F.normalize(model.means.detach() - centre[None], dim=-1))  # [N, K]
            Y2 = (Y * Y)[:, :, None] * u[:, None, :]                                     # [N, K, C]
        return Y2[:, :1], Y2[:, 1:]


def pcg(Aop, b, Minv, iters, rtol):
    """Preconditioned CG on tuples of tensors; returns x and the relative residual reached."""
    dot = lambda a, c: sum(float((x * y).sum()) for x, y in zip(a, c))                  # noqa: E731
    x = tuple(torch.zeros_like(t) for t in b)
    r = tuple(t.clone() for t in b)
    z = tuple(m * t for m, t in zip(Minv, r))
    p = tuple(t.clone() for t in z)
    rz = dot(r, z)
    b_norm = math.sqrt(max(dot(b, b), 1e-30))
    rel = 1.0
    for _ in range(iters):
        Ap = Aop(p)
        alpha = rz / max(dot(p, Ap), 1e-30)
        x = tuple(xi + alpha * pi for xi, pi in zip(x, p))
        r = tuple(ri - alpha * api for ri, api in zip(r, Ap))
        rel = math.sqrt(max(dot(r, r), 0.0)) / b_norm
        if rel < rtol:
            break
        z = tuple(m * t for m, t in zip(Minv, r))
        rz_new = dot(r, z)
        p = tuple(zi + (rz_new / max(rz, 1e-30)) * pi for zi, pi in zip(z, p))
        rz = rz_new
    return x, rel


class LM:
    """The LM phase. Holds the trust region and the line-search gamma across iterations."""

    def __init__(self, model, groups, views_of, targets_of, span, lam, sh_basis, cfg, log=print):
        self.model, self.groups, self.views_of, self.targets_of = model, groups, views_of, targets_of
        self.span, self.lam, self.sh_basis, self.cfg, self.log = span, lam, sh_basis, cfg, log
        self.win = ssim_window(model.means.device)
        self.radius = cfg["radius"]
        self.decrease = 2.0
        self.gamma_max = 0.0
        self.history = []

    def make(self, g):
        return Group(self.model, self.views_of(g), self.span, self.targets_of(g), self.win, self.lam)

    def cost(self, gs):
        """0.5 sum(r^2) over the positions' faces at the current coefficients (= the training loss, up to scale)."""
        with torch.no_grad():
            tot = 0.0
            for g in gs:
                vm, K, W, H = self.views_of(g)
                f = self.model.render_batch(vm, K, W, H, self.span)
                tot += 0.5 * float((residuals(f, self.targets_of(g), self.win, self.lam) ** 2).sum())
        return tot

    def subsets(self, it):
        n, m, k = len(self.groups), self.cfg["subset_positions"], self.cfg["subsets"]
        stride = max(1, n // m)
        out = []
        for s in range(k):
            start = (it * k + s) % stride
            out.append([self.groups[i] for i in range(start, n, stride)][:m])
        return out

    def step(self, it):
        cfg, model = self.cfg, self.model
        t0 = time.time()
        deg = model.sh_degree
        xs, ds, used = [], [], []
        for gs in self.subsets(it):
            G = [self.make(g) for g in gs]
            for Gi in G:
                Gi.linearise()
            b0 = torch.zeros_like(model.params["sh0"]); bN = torch.zeros_like(model.params["shN"])
            d0 = torch.zeros_like(b0); dN = torch.zeros_like(bN)
            for Gi in G:
                g0, gN = Gi.gradient(); b0 += g0; bN += gN
                e0, eN = Gi.diag(self.sh_basis, deg); d0 += e0; dN += eN
            C0, CN = d0.clamp(cfg["min_diag"], cfg["max_diag"]), dN.clamp(cfg["min_diag"], cfg["max_diag"])
            Minv = (1.0 / (d0 + C0), 1.0 / (dN + CN))
            damp = (C0 / self.radius, CN / self.radius)

            def Aop(p):
                a0 = damp[0] * p[0]; aN = damp[1] * p[1]
                for Gi in G:
                    j0, jN = Gi.jtj(p[0], p[1]); a0 = a0 + j0; aN = aN + jN
                return (a0, aN)

            x, rel = pcg(Aop, (b0, bN), Minv, cfg["pcg_iters"], cfg["pcg_rtol"])
            vmax = max(float(x[0].abs().max()), float(x[1].abs().max()), 1e-30)
            s = min(1.0, cfg["max_step"] / vmax)
            xs.append((x[0] * s, x[1] * s)); ds.append((d0, dN)); used.extend(gs)
            last_G, last_gs = G, gs
            self.log(f"    LM {it} subset {len(xs)}: {len(gs)} positions, PCG rel. residual {rel:.2e}")
        # combine the subsets' solutions, weighted by diag(J^T J)
        den0 = sum(d[0] for d in ds) + 1e-12; denN = sum(d[1] for d in ds) + 1e-12
        x0 = sum(xi[0] * di[0] for xi, di in zip(xs, ds)) / den0
        xN = sum(xi[1] * di[1] for xi, di in zip(xs, ds)) / denN
        prev0, prevN = model.params["sh0"].data.clone(), model.params["shN"].data.clone()

        def apply(gamma):
            model.params["sh0"].data.copy_(prev0 - gamma * x0)
            model.params["shN"].data.copy_(prevN - gamma * xN)

        # backtracking line search on a share of the positions used
        rng = np.random.default_rng(it)
        ls = [used[i] for i in rng.permutation(len(used))[:max(1, int(len(used) * cfg["linesearch_frac"]))]]
        gamma = cfg["gamma0"] if self.gamma_max <= 0 else self.gamma_max
        prev = 1e30
        while True:
            apply(gamma)
            c = self.cost(ls)
            if c > prev:
                gamma /= cfg["gamma_alpha"]
                break
            if gamma < 1e-10:
                break
            prev = c
            gamma *= cfg["gamma_alpha"]
        self.gamma_max = max(self.gamma_max, gamma)
        # trust region on the last subset: actual vs linearised cost change
        apply(0.0)
        F_prev = sum(0.5 * float((Gi.r ** 2).sum()) for Gi in last_G)
        F_lin = 0.0
        for Gi in last_G:
            jx = Gi.res_jvp(Gi.Jp(-gamma * x0, -gamma * xN))
            F_lin += 0.5 * float(((Gi.r + jx) ** 2).sum())
        apply(gamma)
        del last_G
        F_new = self.cost(last_gs)
        model_change, change = F_prev - F_lin, F_prev - F_new
        ratio = change / model_change if model_change != 0 else 0.0
        ok = change > 0 and ratio > cfg["min_rel_decrease"]
        if ok:
            self.radius = min(max(self.radius / max(1 / 3, 1 - (2 * ratio - 1) ** 3), cfg["radius_min"]), cfg["radius_max"])
            self.decrease = 2.0
        else:
            self.radius /= self.decrease
            if change < 0:
                apply(0.0)                                   # undo
                self.decrease *= 2
            self.radius = max(self.radius, cfg["radius_min"])
        torch.cuda.synchronize()
        rec = {"lm_iteration": it, "gamma": gamma, "accepted": bool(change > 0), "cost_prev": F_prev, "cost_new": F_new,
               "cost_linearised": F_lin, "ratio": ratio, "radius": self.radius, "seconds": time.time() - t0,
               "positions_used": len(used)}
        self.history.append(rec)
        self.log(f"  LM {it}: gamma {gamma:.3g}, cost {F_prev:.4g} -> {F_new:.4g} (linearised {F_lin:.4g}), "
                 f"{'accepted' if change > 0 else 'undone'}, radius {self.radius:.2e}, {rec['seconds']:.1f} s")
        return rec

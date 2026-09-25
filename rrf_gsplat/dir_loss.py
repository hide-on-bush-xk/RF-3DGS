"""The main-peak direction loss (docs/stage2_notes.md "协议 v1" §16, plan item 3).

The pixel loss plateaus while the main peak is still wrong: the models fit only ~50 % of their own training views'
distinct main peaks within 1 deg (§16). The selection metrics are choices (steer the beam to the predicted maximum),
flat almost everywhere; their relaxation is an expectation under a softmax policy over directions, a convex
combination of per-direction rewards, which is differentiable. In simulation every direction's reward is known from
the target, so the expectation is computed exactly (no sampling, i.e. not RL).

Over the four faces of one receiver position (the full horizontal ring, so the peak may be on any face):

  pi_i  = softmax_i( pred_dB_i / T + log dOmega_i )       a distribution over directions: its density in solid angle
                                                          is exp(pred_dB / T); dOmega_i = cos^3(theta_i) / f^2 is the
                                                          pinhole pixel's solid angle (without it the corners, which
                                                          cover less sky per pixel, would weigh the same)
  g_i   = 10^((true_dB_i - max true_dB) / 10)             the true power arriving from direction i relative to the
                                                          strongest direction: the gain of a pencil beam steered there
  expgain:  L = -10 log10( sum_i pi_i g_i )               the expected beam-gain loss in dB (0 when all of pi sits on
                                                          the true maximum; as T -> 0 it is the loss of steering to the
                                                          predicted maximum, i.e. the metric itself)
  ce:       L = KL(q || pi), q = softmax(true_dB / T + log dOmega)   (the target's own softmax; 0 at pred = truth)

The dB form rather than 1 - sum(pi g): while pi is spread out, sum(pi g) is small and the linear form's gradient
vanishes with it; the log normalises it (d L / d logit_i ~ pi_i (1 - g_i / E[g])).

Returned in the value channel's normalised units (dB / span), like the pixel L1 it is added to.
"""

from __future__ import annotations

import math

import torch

LN10_OVER_10 = math.log(10.0) / 10.0


def solid_angle_logw(K, width, height, device=None):
    """log dOmega per pixel [H, W] of a pinhole camera, up to a constant: -1.5 log(1 + u^2 + v^2)."""
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u = (torch.arange(width, device=device, dtype=torch.float32) + 0.5 - cx) / fx
    v = (torch.arange(height, device=device, dtype=torch.float32) + 0.5 - cy) / fy
    return -1.5 * torch.log1p(u[None, :] ** 2 + v[:, None] ** 2)


def dir_loss(pred, gt, logw, span_db, temp_db, kind="expgain"):
    """pred, gt [F, H, W] (the F faces of one receiver position) in normalised units (dB above the floor / span_db);
    logw [H, W] from solid_angle_logw. Returns a scalar in dB / span_db (expgain) or nats (ce)."""
    lw = logw.reshape(1, -1).expand(pred.shape[0], -1).reshape(-1)
    pdb = pred.reshape(-1) * span_db
    gdb = gt.reshape(-1).to(pdb.dtype) * span_db
    logpi = torch.log_softmax(pdb / temp_db + lw, 0)
    if kind == "expgain":
        ln_g = (gdb - gdb.max()) * LN10_OVER_10
        ln_e = torch.logsumexp(logpi + ln_g, 0)                       # ln E_pi[g] <= 0
        return -ln_e / LN10_OVER_10 / span_db
    if kind == "ce":
        logq = torch.log_softmax(gdb / temp_db + lw, 0)
        return (logq.exp() * (logq - logpi)).sum()
    raise ValueError(f"unknown direction loss {kind!r}")

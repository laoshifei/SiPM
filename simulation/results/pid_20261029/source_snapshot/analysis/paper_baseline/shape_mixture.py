"""Shared Gaussian shape cores with per-source proportions and a broad Cauchy term.

Components are ordered L/H by score. They are not alpha/gamma truth labels.
The Cauchy nuisance density has full real support, so no tail score cut is needed.
"""
from __future__ import annotations

import math
import numpy as np


def densities(x, model):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(model["means"])
    sigma = np.asarray(model["sigmas"])
    normal = np.exp(-0.5 * ((x[:, None] - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
    z = (x - model["background_center"]) / model["background_scale"]
    background = 1 / (np.pi * model["background_scale"] * (1 + z*z))
    return np.column_stack([normal, background])


def fit_mixture(x, group, cores=2, initial=None, background_scale_factor=1.0,
                max_iter=400, tolerance=1e-7):
    x = np.asarray(x, dtype=float)
    groups, group_index = np.unique(group, return_inverse=True)
    if not np.all(np.isfinite(x)) or len(x) < 20 or np.ptp(x) <= 0:
        raise ValueError("Mixture fit requires at least 20 finite, nonconstant scores")
    q05, q95 = np.quantile(x, [0.05, 0.95])
    scale = max(float(q95-q05), float(np.std(x)) * 0.05, 1e-8)
    floor = scale * 0.01
    background = {"background_center": float((q05 + q95) / 2),
                  "background_scale": scale * background_scale_factor,
                  "sigma_floor": floor}
    if initial is not None:
        seeds = [np.asarray(initial["means"], dtype=float)]
        background = {key: initial[key] for key in background}
        floor = background["sigma_floor"]
    elif cores == 1:
        seeds = [np.array([np.median(x)]), np.array([np.quantile(x, 0.25)]), np.array([np.quantile(x, 0.75)])]
    else:
        seeds = [np.quantile(x, [0.20, 0.80]), np.quantile(x, [0.35, 0.65])]
        if len(groups) == 2:
            seeds.insert(0, np.sort([np.median(x[group_index == j]) for j in range(2)]))
    best = None
    for seed in seeds:
        model = {**background, "means": seed.tolist(),
                 "sigmas": [max(scale / (4 * cores), floor)] * cores}
        weights = np.full((len(groups), cores + 1), 0.95 / cores)
        weights[:, -1] = 0.05
        if initial is not None:
            model["sigmas"] = list(initial["sigmas"])
            weights = np.asarray(initial["weights"], dtype=float).copy()
        previous = -np.inf
        history = []
        converged = False
        for iteration in range(max_iter):
            weighted = densities(x, model) * weights[group_index]
            total = np.maximum(weighted.sum(axis=1), 1e-300)
            ll = float(np.log(total).sum())
            history.append(ll)
            responsibility = weighted / total[:, None]
            if iteration and abs(ll - previous) / len(x) < tolerance:
                converged = True
                break
            previous = ll
            totals = responsibility[:, :cores].sum(axis=0)
            means = (responsibility[:, :cores] * x[:, None]).sum(axis=0) / np.maximum(totals, 1e-100)
            variance = (responsibility[:, :cores] * (x[:, None] - means)**2).sum(axis=0) / np.maximum(totals, 1e-100)
            order = np.argsort(means)
            model["means"] = means[order].tolist()
            model["sigmas"] = np.maximum(np.sqrt(variance[order]), floor).tolist()
            ordering = list(order) + [cores]
            for j in range(len(groups)):
                weights[j] = np.maximum(responsibility[group_index == j].mean(axis=0)[ordering], 1e-12)
                weights[j] /= weights[j].sum()
        final_density = densities(x, model) * weights[group_index]
        ll = float(np.log(np.maximum(final_density.sum(axis=1), 1e-300)).sum())
        parameters = 2 * cores + len(groups) * cores
        model.update({"weights": weights.tolist(), "groups": [int(g) for g in groups],
                      "log_likelihood": ll, "bic": parameters * math.log(len(x)) - 2 * ll,
                      "free_parameters": parameters, "n": len(x), "converged": converged,
                      "iterations": iteration + 1, "at_sigma_floor": bool(np.any(np.asarray(model["sigmas"]) <= floor * 1.01)),
                      "history_min_step": float(np.min(np.diff(history))) if len(history) > 1 else None,
                      "background_policy": "Fixed broad Cauchy nuisance; center/scale estimated once from training data"})
        if cores == 2:
            model["shape_fom_fwhm"] = (model["means"][1] - model["means"][0]) / (2 * np.sqrt(2 * np.log(2)) * sum(model["sigmas"]))
        if best is None or model["log_likelihood"] > best["log_likelihood"]:
            best = model
    return best


def fit_weights(x, model, initial=None):
    d = densities(x, model)
    w = np.asarray(initial if initial is not None else [1 / d.shape[1]] * d.shape[1], dtype=float)
    for iteration in range(500):
        weighted = d * w
        posterior = weighted / np.maximum(weighted.sum(axis=1), 1e-300)[:, None]
        updated = np.maximum(posterior.mean(axis=0), 1e-12)
        updated /= updated.sum()
        if np.max(np.abs(updated - w)) < 1e-9:
            w = updated
            break
        w = updated
    return w


def cdf(values, model, weights):
    values = np.asarray(values)
    result = np.zeros(len(values))
    for mu, sigma, weight in zip(model["means"], model["sigmas"], weights[:-1]):
        z = (values - mu) / (sigma * np.sqrt(2))
        result += weight * (1 + np.array([math.erf(v) for v in z])) / 2
    result += weights[-1] * (0.5 + np.arctan((values - model["background_center"]) / model["background_scale"]) / np.pi)
    return result


def diagnostics(x, model, weights):
    low = min(mu - 5*sigma for mu, sigma in zip(model["means"], model["sigmas"]))
    high = max(mu + 5*sigma for mu, sigma in zip(model["means"], model["sigmas"]))
    edges = np.r_[-np.inf, np.linspace(low, high, 61), np.inf]
    counts, _ = np.histogram(x, edges)
    predicted = np.maximum(np.diff(cdf(edges, model, weights)), 0)
    predicted /= predicted.sum()
    return {"mean_log_density": float(np.log(np.maximum(densities(x, model) @ weights, 1e-300)).mean()),
            "binned_total_variation": float(np.abs(counts / len(x) - predicted).sum() / 2),
            "score_edges": edges[1:-1].tolist(), "counts_with_flow": counts.tolist(),
            "expected_with_flow": (predicted * len(x)).tolist()}

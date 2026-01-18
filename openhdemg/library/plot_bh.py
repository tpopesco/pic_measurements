import openhdemg.library as emg
from openhdemg.library.openfiles import emg_from_demuse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from openhdemg.library.tools import delete_empty_mus
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def load_emg_file(file_path):
    emgfile = emg_from_demuse(filepath=file_path)
    emgfile = delete_empty_mus(emgfile)
    good_indices = []
    for idx, mu in enumerate(emgfile["BINARY_MUS_FIRING"]):
        if np.sum(mu) >= 5:
            good_indices.append(idx)
    if not good_indices:
        print("No good MUs (>=5 spikes) found. Skipping.")
        return emgfile, None
    emgfile["BINARY_MUS_FIRING"] = [emgfile["BINARY_MUS_FIRING"][i] for i in good_indices]
    try:
        smoothfits = emg.compute_svr(emgfile=emgfile, discontfiring_dur=1.0)["gensvr"]
    except Exception as e:
        print(f"SVR smoothing failed: {e}")
        smoothfits = None
    return emgfile, smoothfits


def plot_brace_height(emgfile, smoothfits, mu_id, file_path):
    ref_signal = np.array(emgfile["REF_SIGNAL"]).flatten()
    ref_signal *= -1

    baseline_force = np.mean(ref_signal[:4096])
    peak_force_val = np.max(ref_signal)
    force_percent_mvc = (ref_signal - baseline_force) / (peak_force_val - baseline_force) * 20

    firing_times = np.where(emgfile["BINARY_MUS_FIRING"][mu_id] == 1)[0]
    if len(firing_times) < 5:
        print(f"MU {mu_id}: Not enough spikes.")
        return

    t_rec = firing_times[0]
    t_peak = t_rec + np.nanargmax(smoothfits[mu_id][t_rec:])

    if t_peak - t_rec < 20:
        print(f"MU {mu_id}: Duration too short between recruitment and peak.")
        return

    forces = force_percent_mvc[t_rec:t_peak]
    rates = smoothfits[mu_id][t_rec:t_peak]

    # 🛠️ New check: skip if smoothing is broken or mostly NaN
    if np.all(np.isnan(rates)) or np.sum(~np.isnan(rates)) < 3:
        print(f"MU {mu_id}: Smoothing contains only NaNs or too few valid points.")
        return

    # Optional: remove NaNs from rates & matching force values
    valid = ~np.isnan(rates)
    forces = forces[valid]
    rates = rates[valid]

    unique_forces, unique_indices = np.unique(forces, return_index=True)
    unique_rates = rates[unique_indices]
    if len(unique_forces) < 3:
        print(f"MU {mu_id}: Not enough unique force points.")
        return

    interp_f = np.linspace(unique_forces[0], unique_forces[-1], 1000)
    interp_func = interp1d(unique_forces, unique_rates, kind='cubic', fill_value="extrapolate")
    interp_rates = interp_func(interp_f)

    f1, f2 = interp_f[0], interp_f[-1]
    y1 = np.percentile(interp_rates[:5], 25)
    y2 = interp_rates[-1]
    hypo_vec = np.array([f2 - f1, y2 - y1])
    hypo_len = np.linalg.norm(hypo_vec)
    if hypo_len == 0:
        print(f"MU {mu_id}: Hypotenuse length is zero.")
        return

    hypo_unit = hypo_vec / hypo_len
    perp_unit = np.array([-hypo_unit[1], hypo_unit[0]])

    point_vecs = np.stack((interp_f - f1, interp_rates - y1), axis=-1)
    projections = np.dot(point_vecs, hypo_unit)
    closest_points = np.outer(projections, hypo_unit)
    ortho_vecs = point_vecs - closest_points

    signed_ortho_dists = np.einsum('ij,j->i', ortho_vecs, perp_unit)
    #valid_indices = np.where(signed_ortho_dists > 0)[0]
    #if len(valid_indices) == 0:
    #    print(f"MU {mu_id}: No brace height upward.")
    #    return

    #max_dist_idx = valid_indices[np.argmax(signed_ortho_dists[valid_indices])]
    max_dist_idx = np.argmax(signed_ortho_dists)
    x0 = interp_f[max_dist_idx]
    y0 = interp_rates[max_dist_idx]
    proj_point = np.array([f1, y1]) + closest_points[max_dist_idx]
    bh_vec = np.array([x0, y0]) - proj_point
    bh_raw = np.linalg.norm(bh_vec)

    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(interp_f, interp_rates, color="black", linewidth=2, label=f"MU {mu_id} Smoothed DR", antialiased=True)
    plt.plot([f1, f2], [y1, y2], '--', color="gray", linewidth=2)
    plt.scatter(x0, y0, color="red", zorder=5)
    plt.scatter(proj_point[0], proj_point[1], color="blue", zorder=6)
    plt.plot([x0, proj_point[0]], [y0, proj_point[1]], color="red", linestyle="--", linewidth=2, label="Brace height")

    plt.title(f"Brace Height – MU {mu_id}")
    plt.xlabel("Force (%MVC)")
    plt.ylabel("Discharge Rate (pps)")
    plt.legend()
    plt.xlim(min(interp_f) - 1, max(interp_f) + 1)
    plt.ylim(min(interp_rates) - 1, max(interp_rates) + 1)
    plt.show()

    print(f"MU {mu_id}: Bh_raw = {bh_raw:.2f}")


# ----------------------------
# ✅ USER INPUT SECTION
# ----------------------------

file_path = "C:/Users/tpopesco/OneDrive - Université de Lausanne/Bureau/FNS/FirRate_DeltaF/MnHyperex2024/Cramps/Ramps/prepost/processed_DoHa_GMamazing36_pre.mat"

emg_data, smoothfits = load_emg_file(file_path)

if smoothfits is not None:
    for mu_id in range(len(smoothfits)):
        try:
            plot_brace_height(emg_data, smoothfits, mu_id=mu_id, file_path=file_path)
        except (IndexError, KeyError, TypeError) as e:
            print(f"Skipping MU {mu_id}: {e}")
else:
    print("Skipping plot: no smoothing available.")

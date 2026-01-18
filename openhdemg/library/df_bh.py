import pandas as pd
import numpy as np


def compute_braceheight(
    emgfile,
    smoothfits,
    file_path,
    average_method="all",
    normalisation="False",
    recruitment_difference_cutoff=1.0,
    corr_cutoff=0.7,
    controlunitmodulation_cutoff=0.5,
    clean=True,
):

    bh_raw_values = []
    bh_norm_values = []
    max_rate_values = []
    accel_slope_values = []
    atten_slope_values = []
    angle_deg_values = []
    rec_thresh_values = []
    derec_thresh_values = []
    t_peak_values = []
    ssd_values = []

    ref_signal = np.array(emgfile["REF_SIGNAL"]).flatten()
    fsamp = np.array(emgfile["FSAMP"]).flatten()
    idx_3s = int(3 * fsamp)
    idx_15s = int(15 * fsamp)
    if ref_signal[idx_3s] > ref_signal[idx_15s]:
        ref_signal *= -1
    baseline_force = np.mean(ref_signal[:4096])
    peak_force_val = np.max(ref_signal)
    force_percent_mvc = (ref_signal - baseline_force) / (peak_force_val - baseline_force) * 20

    firing_array = emgfile["BINARY_MUS_FIRING"]
    if isinstance(firing_array, np.ndarray) and firing_array.dtype == object:
        firing_array = firing_array.tolist()

        print("BINARY_MUS_FIRING type:", type(emgfile["BINARY_MUS_FIRING"]))
        print("BINARY_MUS_FIRING[0] type:", type(emgfile["BINARY_MUS_FIRING"][0]))
        print("BINARY_MUS_FIRING[0] value:", emgfile["BINARY_MUS_FIRING"][0])


    for mu_id in range(len(smoothfits)):
        try:
            firing_row = firing_array[mu_id]
            if isinstance(firing_row, np.ndarray) and firing_row.ndim > 1:
                firing_row = firing_row.flatten()
            firing_times = np.where(firing_row == 1)[0]
            if len(firing_times) < 5:
                raise ValueError("Too few spikes")

            t_rec = firing_times[0]
            t_derec = firing_times[-1]
            rec_force = force_percent_mvc[t_rec]
            derec_force = force_percent_mvc[t_derec]

            smooth_len = len(smoothfits[mu_id])
            force_len = len(force_percent_mvc)
            if t_rec >= smooth_len or t_rec >= force_len:
                raise IndexError("t_rec out of bounds")

            segment = smoothfits[mu_id][t_rec:]
            if segment.size == 0 or np.all(np.isnan(segment)):
                raise ValueError("All NaN segment")

            t_peak = t_rec + np.nanargmax(segment)
            if t_peak >= smooth_len or t_peak >= force_len or (t_peak - t_rec < 20):
                raise ValueError("Invalid t_peak")

            rates = smoothfits[mu_id][t_rec:t_peak]
            forces = force_percent_mvc[t_rec:t_peak]

            if np.all(np.isnan(rates)) or np.sum(~np.isnan(rates)) < 3:
                raise ValueError("Too few valid rate points")

            valid = ~np.isnan(rates)
            forces = forces[valid]
            rates = rates[valid]

            unique_forces, unique_indices = np.unique(forces, return_index=True)
            unique_rates = rates[unique_indices]
            if len(unique_forces) < 3:
                raise ValueError("Too few unique force values")

            interp_f = unique_forces
            interp_rates = unique_rates

            f1, f2 = interp_f[0], interp_f[-1]
            y1 = np.percentile(interp_rates[:5], 25)
            y2 = interp_rates[-1]
            hypo_vec = np.array([f2 - f1, y2 - y1])
            hypo_len = np.linalg.norm(hypo_vec)
            if hypo_len == 0:
                raise ValueError("Zero-length hypotenuse")

            hypo_unit = hypo_vec / hypo_len
            perp_unit = np.array([-hypo_unit[1], hypo_unit[0]])

            point_vecs = np.stack((interp_f - f1, interp_rates - y1), axis=-1)
            projections = np.dot(point_vecs, hypo_unit)
            closest_points = np.outer(projections, hypo_unit)
            ortho_vecs = point_vecs - closest_points

            signed_ortho_dists = np.einsum('ij,j->i', ortho_vecs, perp_unit)
            max_dist_idx = np.argmax(signed_ortho_dists)
            x0 = interp_f[max_dist_idx]
            y0 = interp_rates[max_dist_idx]
            proj_point = np.array([f1, y1]) + closest_points[max_dist_idx]
            bh_vec = np.array([x0, y0]) - proj_point
            bh_raw = np.linalg.norm(bh_vec)

            norm_ref_point = np.array([f1, y2])
            vec_to_line = norm_ref_point - np.array([f1, y1])
            proj_len = np.dot(vec_to_line, hypo_unit)
            intersection_point = np.array([f1, y1]) + proj_len * hypo_unit
            green_vec_len = np.linalg.norm(intersection_point - norm_ref_point)
            bh_norm = (bh_raw / green_vec_len) * 100 if green_vec_len != 0 else np.nan

            accel_slope = (y0 - y1) / (x0 - f1) if (x0 - f1) != 0 else np.nan
            atten_slope = (y2 - y0) / (f2 - x0) if (f2 - x0) != 0 else np.nan
            v1 = np.array([x0 - f1, y0 - y1])
            v2 = np.array([f2 - x0, y2 - y0])
            cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
            internal_angle = np.degrees(np.arccos(cos_theta))
            angle_deg = 180 + internal_angle

        
            t_descend = t_derec - t_peak
            t_ascend = t_peak - t_rec
            ssd = (t_descend - t_ascend)/(t_descend + t_ascend)*100
            if  t_peak - t_rec < 20:
                raise ValueError("Invalid ssd")

            # Save results
            bh_raw_values.append(bh_raw)
            bh_norm_values.append(bh_norm)
            max_rate_values.append(y2)
            accel_slope_values.append(accel_slope)
            atten_slope_values.append(atten_slope)
            angle_deg_values.append(angle_deg)
            rec_thresh_values.append(rec_force)
            derec_thresh_values.append(derec_force)
            t_peak_values.append(t_peak)
            ssd_values.append(ssd)

        except Exception as e:
            # On any failure, append NaNs
            bh_raw_values.append(np.nan)
            bh_norm_values.append(np.nan)
            max_rate_values.append(np.nan)
            accel_slope_values.append(np.nan)
            atten_slope_values.append(np.nan)
            angle_deg_values.append(np.nan)
            rec_thresh_values.append(np.nan)
            derec_thresh_values.append(np.nan)
            t_peak_values.append(np.nan)
            ssd_values.append(np.nan)

    return pd.DataFrame({
        "MU": list(range(len(bh_raw_values))),
        "Bh raw": bh_raw_values,
        "Bh norm": bh_norm_values,
        "MaxRate": max_rate_values,
        "AccelSlope": accel_slope_values,
        "AttenSlope": atten_slope_values,
        "AngleDeg": angle_deg_values,
        "RecThresh": rec_thresh_values,
        "DerecThresh": derec_thresh_values,
        "Peak": t_peak_values,
        "SSD": ssd_values
    })

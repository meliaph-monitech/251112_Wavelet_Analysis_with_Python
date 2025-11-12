# app.py
import streamlit as st
import pandas as pd
import numpy as np
import zipfile, os, json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Wavelet libs
import pywt
from ssqueezepy import cwt as ss_cwt, ssq_cwt
from ssqueezepy import Wavelet as SSqWavelet
from pycwt import wavelet as pycwt_wavelet

# Optional (for PNG export). If missing, PNG download will show an info message.
import base64

st.set_page_config(layout="wide", page_title="Wavelet Analysis Lab")

# =====================================================
# ---- 1) Bead Segmentation (unchanged from your app)
# =====================================================
def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

# =====================================================
# ---- 2) Session State
# =====================================================
if "segmented_data" not in st.session_state:
    st.session_state.segmented_data = None
if "observations" not in st.session_state:
    st.session_state.observations = []  # list of dicts: csv, bead, signal, status, data(pd.Series)

# =====================================================
# ---- 3) Sidebar: Step 1 (Upload & Segmentation)
# =====================================================
st.sidebar.header("Step 1: Upload & Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV files", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [n for n in zip_ref.namelist() if n.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)
    columns = sample_df.columns.tolist()
    seg_col = st.sidebar.selectbox("Column for Segmentation", columns)
    seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=1.0)
    segment_btn = st.sidebar.button("Bead Segmentation")

if uploaded_zip and 'segment_btn' in locals() and segment_btn:
    segmented_data = {}
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.csv'):
                with zip_ref.open(file_name) as f:
                    df = pd.read_csv(f)
                bead_ranges = segment_beads(df, seg_col, seg_thresh)
                bead_dict = {}
                for idx, (start, end) in enumerate(bead_ranges, start=1):
                    bead_dict[idx] = df.iloc[start:end+1].reset_index(drop=True)
                segmented_data[os.path.basename(file_name)] = bead_dict
    st.session_state.segmented_data = segmented_data
    st.session_state.observations.clear()
    st.success("✅ Bead segmentation complete and locked!")

# =====================================================
# ---- 4) Sidebar: Step 2 (Add Data)
# =====================================================
if st.session_state.segmented_data:
    st.sidebar.header("Step 2: Add Data for Analysis")
    selected_csv = st.sidebar.selectbox("Select CSV File", list(st.session_state.segmented_data.keys()))
    available_beads = list(st.session_state.segmented_data[selected_csv].keys())
    selected_bead = st.sidebar.selectbox("Select Bead Number", available_beads)

    bead_df = st.session_state.segmented_data[selected_csv][selected_bead]
    signal_col = st.sidebar.selectbox("Select Signal Column", bead_df.columns.tolist())
    status = st.sidebar.selectbox("Weld Status", ["OK", "NOK"])

    if st.sidebar.button("➕ Add Data"):
        st.session_state.observations.append({
            "csv": selected_csv,
            "bead": selected_bead,
            "signal": signal_col,
            "status": status,
            "data": bead_df[signal_col].reset_index(drop=True)
        })

    if st.sidebar.button("🔄 Reset Analysis (keep segmentation)"):
        st.session_state.observations.clear()

# =====================================================
# ---- 5) Helpers
# =====================================================
def _color(status): return "green" if status == "OK" else "red"

def normalize_mag(Z, mode, db_range=None):
    """
        Z: 2D nonnegative magnitude array (float)
        mode: "Auto" | "% of Max" | "Fixed dB"
        db_range: (min_db, max_db) if Fixed dB
    """
    eps = 1e-12
    Z = np.asarray(Z, dtype=float)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "Auto":
        Zdb = 20.0 * np.log10(np.maximum(Z, eps))
        Zdb = np.nan_to_num(Zdb, nan=0.0, posinf=0.0, neginf=0.0)
        return Zdb, None, None
    elif mode == "% of Max":
        m = float(max(Z.max(), eps))
        Zp = 100.0 * (Z / m)
        return Zp, 0.0, 100.0
    elif mode == "Fixed dB":
        Zdb = 20.0 * np.log10(np.maximum(Z, eps))
        zmin, zmax = db_range if db_range is not None else (-120.0, 0.0)
        return Zdb, float(zmin), float(zmax)
    else:
        Zdb = 20.0 * np.log10(np.maximum(Z, eps))
        return Zdb, None, None

def safe_heatmap(fig, Z_disp, t, freqs, title, zmin=None, zmax=None, colorscale="Viridis", name=None):
    Z_disp = np.asarray(Z_disp, dtype=float)
    t = np.asarray(t, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    Z_disp = np.nan_to_num(Z_disp, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    freqs = np.nan_to_num(freqs, nan=0.0, posinf=0.0, neginf=0.0)

    hm = go.Heatmap(
        z=Z_disp.tolist(),
        x=t.tolist(),
        y=freqs.tolist(),
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(title="Magnitude"),
        name=name or ""
    )
    fig.add_trace(hm)
    fig.update_layout(title=title, xaxis_title="Time [s]", yaxis_title="Frequency [Hz]")

def _cwt_freqs_from_scales(scales, fs):
    # Using PyWavelets' scale2frequency with Morlet ('morl') convention
    return pywt.scale2frequency('morl', np.asarray(scales, dtype=float)) * float(fs)

def _download_fig_button(fig, filename_prefix):
    try:
        png_bytes = fig.to_image(format="png", engine="kaleido")
        st.download_button("📥 Download PNG", data=png_bytes,
                           file_name=f"{filename_prefix}.png", mime="image/png")
    except Exception:
        st.info("Install Kaleido (`pip install kaleido`) to enable PNG download.")

def _download_csv_button(df_or_dict, filename_prefix):
    if isinstance(df_or_dict, dict):
        try:
            out_df = pd.DataFrame(df_or_dict)
            csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv,
                               file_name=f"{filename_prefix}.csv", mime="text/csv")
        except Exception:
            payload = json.dumps({k: np.asarray(v).tolist() for k, v in df_or_dict.items()})
            st.download_button("📥 Download JSON", data=payload,
                               file_name=f"{filename_prefix}.json", mime="application/json")
    else:
        csv = df_or_dict.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv,
                           file_name=f"{filename_prefix}.csv", mime="text/csv")

def _auto_trim_by_bead(observations):
    """Group obs by bead and trim signals within each bead to the shortest length."""
    groups = {}
    for obs in observations:
        groups.setdefault(obs["bead"], []).append(obs)
    out = {}
    for bead, obs_list in groups.items():
        min_len = min(len(o["data"]) for o in obs_list)
        out[bead] = [(o, o["data"].iloc[:min_len].reset_index(drop=True)) for o in obs_list]
    return out

def _pairwise_trim(a: pd.Series, b: pd.Series):
    n = min(len(a), len(b))
    return a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)

# COI helper using pycwt just to compute coi curve (Morlet)
def _compute_coi_curve(x, fs, morlet_omega0=6):
    """
    Returns time vector t and frequency COI curve freq_coi (Hz) for a signal x at fs.
    We only use pycwt to get coi for Morlet wavelet.
    """
    dt = 1.0 / float(fs)
    mother = pycwt_wavelet.Morlet(morlet_omega0)
    # Use pycwt.cwt to get coi and period
    # pycwt expects demeaned series
    x0 = np.asarray(x, dtype=float)
    x0 = x0 - np.mean(x0)
    # Minimal params: scales will be auto; we only need coi + period
    W, scales, freqs, coi, fft, ff, cdelta, *_ = pycwt_wavelet.cwt(x0, dt, mother)  # freqs=1/period
    t = np.arange(len(x0)) * dt
    freq_coi = 1.0 / np.asarray(coi, dtype=float)
    freq_coi = np.nan_to_num(freq_coi, nan=0.0, posinf=0.0, neginf=0.0)
    return t, freq_coi

# =====================================================
# ---- 6) Main: Wavelet Analysis Tabs
# =====================================================
if st.session_state.observations:
    st.title("Wavelet Analysis")

    tabs = st.tabs([
        "CWT (Scalogram)",
        "Synchrosqueezed CWT",
        "DWT Denoise + MRA",
        "MODWT + MRA",
        "Wavelet Packets",
        "Wavelet Coherence"
    ])

    # -------------------------------------------------
    # Tab 1: CWT
    # -------------------------------------------------
    with tabs[0]:
        fs = st.number_input("Sampling Rate (Hz) — CWT", value=10000, min_value=1)
        norm_mode = st.selectbox("Normalization — CWT", ["Auto", "% of Max", "Fixed dB"])
        dbrange = st.slider("dB Range — CWT", -140, 0, (-100, -20)) if norm_mode == "Fixed dB" else None
        show_coi = st.checkbox("Show COI (cone of influence)", value=True)

        with st.expander("Advanced (CWT)"):
            voices_per_octave = st.slider("Voices/Octave (resolution)", 16, 64, 32, step=8)
            morlet_mu = st.slider("Morlet μ", 3, 12, 6)

        fig = go.Figure()
        # Group by bead and auto-trim inside each bead for overlay plots
        by_bead = _auto_trim_by_bead(st.session_state.observations)
        for bead, items in by_bead.items():
            for obs, s in items:
                x = s.to_numpy(dtype=float)
                # ssqueezepy CWT (stable unpacking)
                Wx, scales = ss_cwt(x, wavelet=SSqWavelet(('morlet', {'mu': morlet_mu})),
                                    nv=voices_per_octave)
                freqs = _cwt_freqs_from_scales(scales, fs)
                Z = np.abs(Wx).astype(float)
                Z_disp, zmin, zmax = normalize_mag(Z, norm_mode, dbrange)
                t = np.arange(len(x), dtype=float) / float(fs)

                safe_heatmap(fig, Z_disp, t, freqs,
                             title="CWT (Morlet) — Scalogram",
                             zmin=zmin, zmax=zmax,
                             colorscale="Viridis",
                             name=f"{obs['csv']} · Bead {obs['bead']} ({obs['status']})")

                if show_coi:
                    try:
                        t_coi, f_coi = _compute_coi_curve(x, fs, morlet_omega0=6)
                        fig.add_trace(go.Scatter(
                            x=t_coi.tolist(), y=f_coi.tolist(),
                            mode="lines", line=dict(color="white", width=2, dash="dash"),
                            name="COI"
                        ))
                        # Shade outside COI (lower frequencies under the line)
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([t_coi, t_coi[::-1]]).tolist(),
                            y=np.concatenate([np.zeros_like(f_coi), f_coi[::-1]]).tolist(),
                            fill="toself", mode="lines",
                            line=dict(width=0), fillcolor="rgba(255,255,255,0.15)",
                            name="Outside COI"
                        ))
                    except Exception:
                        st.info("COI overlay unavailable (pycwt not providing coi).")

        st.plotly_chart(fig, use_container_width=True)
        _download_fig_button(fig, "cwt_scalogram")

        # Optional compact CSV: mean magnitude per frequency for each overlay (keeps file small)
        # Build a tidy frame if desired
        if st.checkbox("Prepare CSV (mean over time per frequency)"):
            rows = []
            for bead, items in by_bead.items():
                for obs, s in items:
                    x = s.to_numpy(dtype=float)
                    Wx, scales = ss_cwt(x, wavelet=SSqWavelet(('morlet', {'mu': morlet_mu})),
                                        nv=voices_per_octave)
                    freqs = _cwt_freqs_from_scales(scales, fs)
                    Z = np.abs(Wx).astype(float)
                    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                    mean_mag = Z.mean(axis=1)  # per frequency
                    for fval, mval in zip(freqs, mean_mag):
                        rows.append({
                            "csv": obs["csv"], "bead": obs["bead"], "status": obs["status"],
                            "frequency_hz": float(fval), "mean_magnitude": float(mval)
                        })
            df_out = pd.DataFrame(rows)
            _download_csv_button(df_out, "cwt_mean_per_freq")

    # -------------------------------------------------
    # Tab 2: SSQ-CWT
    # -------------------------------------------------
    with tabs[1]:
        fs = st.number_input("Sampling Rate (Hz) — SSQ", value=10000, min_value=1)
        norm_mode = st.selectbox("Normalization — SSQ", ["Auto", "% of Max", "Fixed dB"])
        dbrange = st.slider("dB Range — SSQ", -140, 0, (-100, -20)) if norm_mode == "Fixed dB" else None
        show_coi = st.checkbox("Show COI (cone of influence) — SSQ", value=True)

        with st.expander("Advanced (SSQ)"):
            voices_per_octave = st.slider("Voices/Octave (resolution) — SSQ", 16, 64, 32, step=8)
            morlet_mu = st.slider("Morlet μ — SSQ", 3, 12, 6)
            show_ridge = st.checkbox("Extract & plot ridge frequency", value=False)

        fig = go.Figure()
        by_bead = _auto_trim_by_bead(st.session_state.observations)
        for bead, items in by_bead.items():
            for obs, s in items:
                x = s.to_numpy(dtype=float)
                # ssqueezepy.ssq_cwt stable unpack
                ssq_out = ssq_cwt(x, fs=float(fs), wavelet=('morlet', {'mu': morlet_mu}),
                                  nv=voices_per_octave)
                Tx = np.asarray(ssq_out[0])             # synchrosqueezed transform
                ssq_freqs = np.asarray(ssq_out[1])      # frequencies (Hz)

                Z = np.abs(Tx).astype(float)
                Z_disp, zmin, zmax = normalize_mag(Z, norm_mode, dbrange)
                t = np.arange(len(x), dtype=float) / float(fs)

                safe_heatmap(fig, Z_disp, t, ssq_freqs,
                             title="Synchrosqueezed CWT — Scalogram",
                             zmin=zmin, zmax=zmax,
                             colorscale="Viridis",
                             name=f"{obs['csv']} · Bead {obs['bead']} ({obs['status']})")

                if show_ridge:
                    ridge_idx = np.argmax(Z, axis=0)
                    ridge_freq = ssq_freqs[ridge_idx]
                    fig.add_trace(go.Scatter(
                        x=t.tolist(), y=ridge_freq.tolist(),
                        mode="lines", line=dict(color="white", width=2),
                        name="Ridge freq"
                    ))

                if show_coi:
                    try:
                        t_coi, f_coi = _compute_coi_curve(x, fs, morlet_omega0=6)
                        fig.add_trace(go.Scatter(
                            x=t_coi.tolist(), y=f_coi.tolist(),
                            mode="lines", line=dict(color="white", width=2, dash="dash"),
                            name="COI"
                        ))
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([t_coi, t_coi[::-1]]).tolist(),
                            y=np.concatenate([np.zeros_like(f_coi), f_coi[::-1]]).tolist(),
                            fill="toself", mode="lines",
                            line=dict(width=0), fillcolor="rgba(255,255,255,0.15)",
                            name="Outside COI"
                        ))
                    except Exception:
                        st.info("COI overlay unavailable (pycwt not providing coi).")

        st.plotly_chart(fig, use_container_width=True)
        _download_fig_button(fig, "ssq_cwt")

    # -------------------------------------------------
    # Tab 3: DWT Denoise + MRA
    # -------------------------------------------------
    with tabs[2]:
        with st.expander("Advanced (DWT)"):
            wave = st.selectbox("Wavelet family", ["db4", "db6", "sym8", "coif3"], index=0)
            maxlev_cap = st.slider("Max levels cap", 1, 10, 6)

        for obs in st.session_state.observations:
            x = obs["data"].to_numpy(dtype=float)
            maxlev = min(maxlev_cap, pywt.dwt_max_level(len(x), pywt.Wavelet(wave).dec_len))
            coeffs = pywt.wavedec(x, wavelet=wave, level=maxlev, mode="symmetric")

            # VisuShrink soft
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uth = sigma * np.sqrt(2 * np.log(len(x)))
            coeffs_th = [coeffs[0]] + [pywt.threshold(c, value=uth, mode="soft") for c in coeffs[1:]]
            x_dn = pywt.waverec(coeffs_th, wavelet=wave, mode="symmetric")

            # MRA components (approx + details)
            # Reconstruct each detail level separately
            comps = []
            for i in range(1, len(coeffs)):
                ckeep = [np.zeros_like(c) for c in coeffs]
                ckeep[0] = np.zeros_like(coeffs[0])
                ckeep[i] = coeffs[i]
                comps.append(pywt.waverec(ckeep, wavelet=wave, mode="symmetric"))

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=x.tolist(), mode="lines", name="Original", line=dict(color="gray")))
            fig.add_trace(go.Scatter(y=x_dn.tolist(), mode="lines", name="Denoised", line=dict(color=_color(obs["status"]))))
            for i, c in enumerate(comps, 1):
                fig.add_trace(go.Scatter(y=np.asarray(c, dtype=float).tolist(), mode="lines", name=f"D{i}"))
            fig.update_layout(title=f"DWT Denoise + MRA — {obs['csv']} · Bead {obs['bead']} ({obs['status']})",
                              xaxis_title="Index", yaxis_title="Amplitude")
            st.plotly_chart(fig, use_container_width=True)
            _download_fig_button(fig, f"dwt_mra_{obs['csv']}_bead{obs['bead']}")

            # CSV download of denoised + components
            df_out = pd.DataFrame({"original": x, "denoised": x_dn})
            for i, c in enumerate(comps, 1):
                df_out[f"D{i}"] = np.asarray(c, dtype=float)
            _download_csv_button(df_out, f"dwt_mra_{obs['csv']}_bead{obs['bead']}")

    # -------------------------------------------------
    # Tab 4: MODWT + MRA (shift-invariant)
    # -------------------------------------------------
    with tabs[3]:
        with st.expander("Advanced (MODWT)"):
            wave = st.selectbox("Wavelet family (MODWT)", ["db4", "db6", "sym8", "coif3"], index=0)
            levels = st.slider("Levels (MODWT)", 1, 10, 6)

        for obs in st.session_state.observations:
            x = obs["data"].to_numpy(dtype=float)
            coeffs = pywt.modwt(x, wavelet=wave, level=levels)
            mra = pywt.modwt_mra(coeffs)

            fig = go.Figure()
            for i, comp in enumerate(mra, 1):
                fig.add_trace(go.Scatter(y=np.asarray(comp, dtype=float).tolist(),
                                         mode="lines", name=f"Level {i}"))
            fig.update_layout(title=f"MODWT MRA — {obs['csv']} · Bead {obs['bead']} ({obs['status']})",
                              xaxis_title="Index", yaxis_title="Component")
            st.plotly_chart(fig, use_container_width=True)
            _download_fig_button(fig, f"modwt_mra_{obs['csv']}_bead{obs['bead']}")

            df_out = pd.DataFrame({f"L{i}": np.asarray(c, dtype=float) for i, c in enumerate(mra, 1)})
            _download_csv_button(df_out, f"modwt_mra_{obs['csv']}_bead{obs['bead']}")

    # -------------------------------------------------
    # Tab 5: Wavelet Packets
    # -------------------------------------------------
    with tabs[4]:
        with st.expander("Advanced (Packets)"):
            wave = st.selectbox("Wavelet family (packets)", ["db4", "db6", "sym8", "coif3"], index=0)
            maxlevel = st.slider("Max level (packets)", 1, 8, 4)
            criterion = st.selectbox("Best-basis criterion", ["entropy", "shannon", "sure", "logenergy"], index=0)

        for obs in st.session_state.observations:
            x = obs["data"].to_numpy(dtype=float)
            wp = pywt.WaveletPacket(data=x, wavelet=wave, mode='symmetric', maxlevel=maxlevel)
            # Best basis (pywt uses 'entropy' as default; we map some aliases)
            crit = "entropy" if criterion in ["entropy", "shannon"] else criterion
            level_nodes = wp.get_best_level(decomp_level=maxlevel, criterion=crit)
            nodes = [n.path for n in level_nodes]
            energies = [float(np.sum(np.square(n.data))) for n in level_nodes]

            fig = go.Figure(go.Bar(x=nodes, y=energies, marker_color=_color(obs["status"])))
            fig.update_layout(title=f"Wavelet Packet Energies — {obs['csv']} · Bead {obs['bead']} ({obs['status']})",
                              xaxis_title="Node", yaxis_title="Energy")
            st.plotly_chart(fig, use_container_width=True)
            _download_fig_button(fig, f"packets_{obs['csv']}_bead{obs['bead']}")

            df_out = pd.DataFrame({"node": nodes, "energy": energies})
            _download_csv_button(df_out, f"packets_{obs['csv']}_bead{obs['bead']}")

    # -------------------------------------------------
    # Tab 6: Wavelet Coherence
    # -------------------------------------------------
    with tabs[5]:
        if len(st.session_state.observations) < 2:
            st.info("Add at least two signals to compare for coherence.")
        else:
            fs = st.number_input("Sampling Rate (Hz) — Coherence", value=10000, min_value=1)
            idxs = list(range(len(st.session_state.observations)))
            a = st.selectbox("Signal A (index in queue)", idxs, format_func=lambda i: f"{st.session_state.observations[i]['csv']} · Bead {st.session_state.observations[i]['bead']} ({st.session_state.observations[i]['status']})")
            b = st.selectbox("Signal B (index in queue)", idxs, index=min(1, len(idxs)-1),
                             format_func=lambda i: f"{st.session_state.observations[i]['csv']} · Bead {st.session_state.observations[i]['bead']} ({st.session_state.observations[i]['status']})")
            if a == b:
                st.warning("Select two different signals.")
            else:
                obs_a = st.session_state.observations[a]
                obs_b = st.session_state.observations[b]
                x, y = _pairwise_trim(obs_a["data"], obs_b["data"])
                dt = 1.0 / float(fs)
                mother = pycwt_wavelet.Morlet(6)

                # pycwt.wct returns:
                # W12, cross, coi, freq, signif, rsq, period, scale, wcoh, phase
                W12, cross, coi, freq, signif, rsq, period, scale, wcoh, phase = \
                    pycwt_wavelet.wct(np.asarray(x, dtype=float),
                                      np.asarray(y, dtype=float),
                                      dt, mother)

                t = np.arange(len(x), dtype=float) * dt
                rsq = np.asarray(rsq, dtype=float)
                period = np.asarray(period, dtype=float)
                freq = np.asarray(1.0 / period, dtype=float)  # Hz

                fig = go.Figure()
                safe_heatmap(fig, rsq, t, freq, title="Wavelet Coherence (rsq)",
                             zmin=0.0, zmax=1.0, colorscale="Turbo",
                             name=f"A: {obs_a['csv']} · B: {obs_b['csv']}")

                # COI overlay
                try:
                    coi = np.asarray(coi, dtype=float)
                    fcoi = 1.0 / np.maximum(coi, 1e-12)
                    fig.add_trace(go.Scatter(
                        x=t.tolist(), y=fcoi.tolist(),
                        mode="lines", line=dict(color="white", width=2, dash="dash"),
                        name="COI"
                    ))
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([t, t[::-1]]).tolist(),
                        y=np.concatenate([np.zeros_like(fcoi), fcoi[::-1]]).tolist(),
                        fill="toself", mode="lines",
                        line=dict(width=0), fillcolor="rgba(255,255,255,0.15)",
                        name="Outside COI"
                    ))
                except Exception:
                    pass

                st.plotly_chart(fig, use_container_width=True)
                _download_fig_button(fig, f"coherence_{obs_a['csv']}_vs_{obs_b['csv']}")

else:
    st.info("1) Upload ZIP and segment beads. 2) Add one or more signals to analyze in the wavelet tabs.")

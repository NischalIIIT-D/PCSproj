import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import os, sys

# Add the module directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampling_quantization import SamplenQuant, plot_sampling_quantization, plot_spectrum
from line_codes import generate_line_code, plot_line_code, plot_all_line_codes, plot_eye_diagram
from pulse_shaping import apply_pulse_shaping, plot_pulse_shapes, raised_cosine_pulse, plot_eq736
from channel_matched_filter import (awgn_channel, awgn_channel_ebn0,
                                     matched_filter, detect_bits,
                                     calculate_ber, theoretical_ber,
                                     plot_ber_curve, plot_channel_effects)
from eye_diagrams import (plot_all_eye_diagrams, plot_pulse_spectra,
                           plot_eye_with_noise, plot_eye_comparison_with_noise)

# Create output directory
os.makedirs('outputs', exist_ok=True)

# ==============================================================
# ██████████████████████████████████████████████████████████████
#                        PARAMETERS
# ██████████████████████████████████████████████████████████████
# ==============================================================

# --- Source Signal Parameters ---
SIGNAL_DURATION = 1.0          # Duration in seconds
FS_ORIGINAL     = 500.0        # Original (high) sampling rate Hz (like MATLAB td=0.002)
F1              = 1.0          # Frequency of first sinusoid (Hz)
F2              = 3.0          # Frequency of second sinusoid (Hz)
# Signal: g(t) = sin(2π*F1*t) - sin(2π*F2*t)  

# --- Sampling & Quantization Parameters ---
FS_NEW          = 50.0         # FS_ORIGINAL— must be integer multiple of New sampling frequency (Hz) 
NUM_BITS        = 4            # Bits per sample (L = 2^NUM_BITS quantization levels)
                                # 2 (4 levels), 4 (16 levels), 8 (256 levels)

#
#  --- Line Code Parameters ---
BIT_RATE        = 100.0        # Bits per second (bps)
Tb              = 1.0 / BIT_RATE  # Bit period (seconds)
SAMPLES_PER_BIT = 50           # Samples per bit period (controls waveform resolution)
LINE_CODE       = 'nrz_polar'  # Options: 'nrz_polar', 'nrz_onoff', 'bipolar',
                                #          'rz_polar', 'rz_onoff'
AMPLITUDE       = 1.0          # Signal amplitude (Volts)

# --- Pulse Shaping Parameters ---
PULSE_TYPE      = 'rc'         # Options: 'rect', 'sinc', 'rc'
BETA            = 0.5          # Rolloff factor for RC pulse (0 ≤ β ≤ 1)
                                # 0=sinc-like, 0.5=standard, 1=maximum rolloff

# --- Eye Diagram Parameters (Section 7.10 from Lathi) ---
EYE_TAU         = 64           # Samples per symbol (textbook uses 64)
EYE_N_SYMBOLS   = 400          # Number of random symbols (textbook uses 400)
EYE_BETA        = 0.5          # RC rolloff for eye diagram
EYE_TD          = 4            # RC truncation length in symbol periods (textbook uses 4)
EYE_SNR_NOISE   = 10           # SNR (dB) to show noisy eye diagram

# --- Channel (AWGN) Parameters ---
SNR_DB          = 10.0         # Eb/N0 in dB for single-run demo
                                # (Energy-per-bit to Noise density ratio)

# --- BER Curve Parameters (BONUS waterfall) ---
N_BITS_BER      = 100_000      # Number of bits for BER simulation (≥100,000 for accuracy)
SNR_RANGE_DB    = list(range(0, 13, 2))  # [0, 2, 4, 6, 8, 10, 12] dB

# --- Which blocks to run (set True/False to enable/disable) ---
RUN_BLOCK_1_SAMPLING    = True
RUN_BLOCK_2_LINE_CODE   = True
RUN_BLOCK_3_PULSE_SHAPE = True
RUN_BLOCK_3B_EYE_DIAG  = True   # Eye diagrams for all pulses (Section 7.10)
RUN_BLOCK_4_CHANNEL     = True
RUN_BLOCK_5_MATCHED     = True
RUN_BLOCK_6_BER         = True
RUN_BONUS_BER_CURVE     = True   # Takes a while (100k bits × 7 SNR points)

# ==============================================================
# END OF PARAMETERS
# ==============================================================


def print_block(num, title):
    print(f"\n{'='*60}")
    print(f"  BLOCK {num}: {title}")
    print(f"{'='*60}")


# ==============================================================
# BLOCK 1: SAMPLING AND QUANTIZATION
# ==============================================================
if RUN_BLOCK_1_SAMPLING:
    print_block(1, "SAMPLING AND QUANTIZATION")

    # Generate the source signal (same as textbook: 1Hz + 3Hz sinusoids)
    t = np.arange(0, SIGNAL_DURATION, 1.0 / FS_ORIGINAL)
    signal = np.sin(2 * np.pi * F1 * t) - np.sin(2 * np.pi * F2 * t)

    print(f"  Signal: g(t) = sin(2π·{F1}t) - sin(2π·{F2}t)")
    print(f"  Original sampling rate: {FS_ORIGINAL} Hz")
    print(f"  New sampling rate:      {FS_NEW} Hz (Nyquist rate for {F2}Hz = {2*F2}Hz)")
    print(f"  Bits per sample:        {NUM_BITS} → L = {2**NUM_BITS} quantization levels")

    # Run the SamplenQuant module
    (t_sampled, s_sampled, s_quantized, s_zoh,
     bitstream, Delta, SQNR, L) = SamplenQuant(signal, t, FS_ORIGINAL, FS_NEW, NUM_BITS)

    print(f"\n  RESULTS:")
    print(f"  Quantization step size (Δ): {Delta:.4f}")
    print(f"  SQNR: {SQNR:.2f} dB")
    print(f"  Total samples: {len(t_sampled)}")
    print(f"  Total bits: {len(bitstream)}")
    print(f"  First 32 bits of PCM: {bitstream[:32]}")

    # Plot 1a: Time domain
    fig1 = plot_sampling_quantization(
        t, signal, t_sampled, s_sampled, s_quantized, s_zoh,
        SQNR, L, FS_NEW, NUM_BITS,
        title_prefix=f"Block 1 | {NUM_BITS}-bit PCM | "
    )
    fig1.savefig('outputs/block1_sampling_quantization.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Plot 1b: Frequency spectra
    fig1b = plot_spectrum(t, signal, s_zoh, FS_ORIGINAL, FS_NEW,
                           title_prefix="Block 1 | ")
    fig1b.savefig('outputs/block1_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close(fig1b)

    # Compare different quantization levels
    fig1c, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ax, nb in zip(axes, [2, 8]):
        (_, _, _, s_zoh_cmp, _, _, SQNR_cmp, L_cmp) = SamplenQuant(
            signal, t, FS_ORIGINAL, FS_NEW, nb
        )
        ax.plot(t, signal, 'k--', linewidth=1.5, label='Original')
        ax.plot(t[:len(s_zoh_cmp)], s_zoh_cmp, 'b-',
                label=f'{L_cmp}-level PCM ({nb}-bit, SQNR={SQNR_cmp:.1f}dB)')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (sec)'); ax.set_ylabel('Amplitude')
    fig1c.suptitle('PCM with Different Quantization Levels', fontsize=13)
    fig1c.savefig('outputs/block1_quantization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig1c)

    print("\n  ✅ Block 1 plots saved to outputs/")


# ==============================================================
# BLOCK 2: LINE CODES
# ==============================================================
if RUN_BLOCK_2_LINE_CODE:
    print_block(2, "LINE CODES")

    # We use the bitstream from Block 1, or generate random bits
    if RUN_BLOCK_1_SAMPLING:
        bits_for_line_code = bitstream[:200]   # first 200 bits
    else:
        np.random.seed(42)
        bits_for_line_code = np.random.randint(0, 2, 200)

    print(f"  Line code: {LINE_CODE.upper()}")
    print(f"  Bit rate: {BIT_RATE} bps | Tb = {Tb*1000:.1f} ms")
    print(f"  Samples per bit: {SAMPLES_PER_BIT}")
    print(f"  First 20 bits: {bits_for_line_code[:20]}")

    # Generate the chosen line code
    t_lc, waveform_lc = generate_line_code(
        bits_for_line_code, Tb, SAMPLES_PER_BIT, LINE_CODE, AMPLITUDE
    )

    # Plot: single line code
    fig2a = plot_line_code(t_lc, waveform_lc, bits_for_line_code[:30],
                            Tb, LINE_CODE, title_prefix="Block 2 | ")
    fig2a.savefig('outputs/block2_line_code.png', dpi=150, bbox_inches='tight')
    plt.close(fig2a)

    # Plot: all line codes comparison
    fig2b = plot_all_line_codes(bits_for_line_code[:30], Tb, SAMPLES_PER_BIT, AMPLITUDE)
    fig2b.savefig('outputs/block2_all_line_codes.png', dpi=150, bbox_inches='tight')
    plt.close(fig2b)

    # Eye diagram of NRZ line code (before pulse shaping)
    fig2c = plot_eye_diagram(
        waveform_lc, SAMPLES_PER_BIT, num_traces=200,
        title=f"Eye Diagram | {LINE_CODE.upper()} | Rectangular Pulse (No Shaping)"
    )
    fig2c.savefig('outputs/block2_eye_diagram_linecode.png', dpi=150, bbox_inches='tight')
    plt.close(fig2c)

    print("\n  ✅ Block 2 plots saved to outputs/")


# ==============================================================
# BLOCK 3: PULSE SHAPING
# ==============================================================
if RUN_BLOCK_3_PULSE_SHAPE:
    print_block(3, "PULSE SHAPING")

    # Need more bits for a good eye diagram
    np.random.seed(42)
    n_bits_ps = 1000
    bits_ps = np.random.randint(0, 2, n_bits_ps)
    bits_ps_polar = 2 * bits_ps - 1   # convert to ±1

    print(f"  Pulse type: {PULSE_TYPE}" +
          (f" | β={BETA}" if PULSE_TYPE == 'rc' else ""))
    print(f"  Bit rate: {BIT_RATE} bps | Samples/bit: {SAMPLES_PER_BIT}")

    # Apply pulse shaping
    t_ps, waveform_ps, pulse, t_pulse = apply_pulse_shaping(
        bits_ps_polar, Tb, SAMPLES_PER_BIT, PULSE_TYPE, BETA
    )

    # Plot comparison of all pulse shapes
    fig3a = plot_pulse_shapes(Tb, SAMPLES_PER_BIT)
    fig3a.savefig('outputs/block3_pulse_shapes_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig3a)

    # ── Equation 7.36 EXACT — dedicated plot ──────────────────────
    # Rb = BIT_RATE (e.g. 100 bps), but for normalized visualization use Rb=1
    fig3_eq736 = plot_eq736(Rb=1.0, n_periods=6)
    fig3_eq736.savefig('outputs/block3_eq736_raised_cosine.png', dpi=150, bbox_inches='tight')
    plt.close(fig3_eq736)
    print("  ✓ Equation 7.36 plot saved")

    # Eye diagram after pulse shaping (CLEAN — no noise yet)
    fig3b = plot_eye_diagram(
        waveform_ps, SAMPLES_PER_BIT, num_traces=500,
        title=f"Eye Diagram | After {PULSE_TYPE.upper()} Pulse Shaping (β={BETA}) | No Noise"
    )
    fig3b.savefig('outputs/block3_eye_diagram_pulseshaped.png', dpi=150, bbox_inches='tight')
    plt.close(fig3b)

    # Compare eye diagrams for different pulse types
    fig3c, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig3c.suptitle('Eye Diagrams for Different Pulse Types (No Noise)', fontsize=13)
    for ax, ptype, extra in zip(axes, ['rect', 'sinc', 'rc'],
                                 ['', '', f' β={BETA}']):
        _, wf, _, _ = apply_pulse_shaping(bits_ps_polar, Tb, SAMPLES_PER_BIT, ptype, BETA)
        trace_len = 2 * SAMPLES_PER_BIT
        t_eye = np.linspace(0, 2, trace_len)
        n_tr = min(len(wf) // trace_len, 200)
        for i in range(n_tr):
            seg = wf[i*trace_len:(i+1)*trace_len]
            ax.plot(t_eye, seg, 'b-', alpha=0.1, linewidth=0.8)
        ax.set_title(f'{ptype.upper()}{extra}')
        ax.set_xlabel('Time (normalized to Tb)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
    fig3c.savefig('outputs/block3_eye_diagrams_all_pulses.png', dpi=150, bbox_inches='tight')
    plt.close(fig3c)

    print("\n  ✅ Block 3 plots saved to outputs/")


# ==============================================================
# BLOCK 3B: EYE DIAGRAMS — Section 7.10, Page 386 (Lathi)
# Python equivalent of binary_eye.m from textbook
# ==============================================================
if RUN_BLOCK_3B_EYE_DIAG:
    print_block("3B", "EYE DIAGRAMS (Section 7.10 — All Pulse Types)")

    print(f"  Tau (samples/symbol): {EYE_TAU}")
    print(f"  Number of symbols:    {EYE_N_SYMBOLS}")
    print(f"  RC rolloff β:         {EYE_BETA}")
    print(f"  RC truncation Td:     {EYE_TD}")

    # ── Plot 1: All 4 eye diagrams (matches binary_eye.m exactly) ──
    print("  Generating: Binary eye diagrams for all 4 pulse types...")
    fig_eye1 = plot_all_eye_diagrams(
        n_symbols=EYE_N_SYMBOLS,
        Tau=EYE_TAU,
        beta=EYE_BETA,
        Td=EYE_TD,
        title_prefix="Binary Eye Diagrams (Section 7.10)"
    )
    fig_eye1.savefig('outputs/block3b_eye_all_pulses.png', dpi=150, bbox_inches='tight')
    plt.close(fig_eye1)
    print("    ✓ All 4 eye diagrams saved")

    # ── Plot 2: Pulse shapes + spectra ──
    print("  Generating: Pulse spectra comparison (RC vs Sinc)...")
    fig_spec1, fig_spec2 = plot_pulse_spectra(
        Tau=EYE_TAU,
        beta_list=[0.0, 0.25, 0.5, 0.75, 1.0],
        Td=EYE_TD
    )
    fig_spec1.savefig('outputs/block3b_pulse_spectra_all.png', dpi=150, bbox_inches='tight')
    fig_spec2.savefig('outputs/block3b_rc_beta_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig_spec1)
    plt.close(fig_spec2)
    print("    ✓ Pulse spectra saved")

    # ── Plot 3: Noisy eye diagrams (eye closing with noise) ──
    print("  Generating: Eye diagrams under increasing noise...")
    fig_noisy = plot_eye_with_noise(
        Tau=EYE_TAU,
        beta=EYE_BETA,
        Td=EYE_TD,
        n_symbols=EYE_N_SYMBOLS,
        snr_levels=[None, 20, EYE_SNR_NOISE, 4]
    )
    fig_noisy.savefig('outputs/block3b_eye_noise_effect.png', dpi=150, bbox_inches='tight')
    plt.close(fig_noisy)
    print("    ✓ Noisy eye diagrams saved")

    # ── Plot 4: All pulses × clean+noisy (2×4 grid) ──
    print("  Generating: Eye diagrams all pulses with/without noise...")
    fig_combo = plot_eye_comparison_with_noise(
        Tau=EYE_TAU,
        beta=EYE_BETA,
        Td=EYE_TD,
        n_symbols=EYE_N_SYMBOLS,
        snr_db=EYE_SNR_NOISE
    )
    fig_combo.savefig('outputs/block3b_eye_all_pulses_with_noise.png',
                      dpi=150, bbox_inches='tight')
    plt.close(fig_combo)
    print("    ✓ Combined eye diagram grid saved")

    print("\n  ✅ Block 3B plots saved to outputs/")


# ==============================================================
# BLOCK 4: AWGN CHANNEL
# ==============================================================
if RUN_BLOCK_4_CHANNEL:
    print_block(4, "AWGN CHANNEL")

    np.random.seed(42)
    n_bits_ch = 1000
    bits_ch = np.random.randint(0, 2, n_bits_ch)
    bits_ch_polar = 2 * bits_ch - 1

    # Get pulse shaped signal
    t_ch, waveform_ch, pulse_ch, t_pulse_ch = apply_pulse_shaping(
        bits_ch_polar, Tb, SAMPLES_PER_BIT, PULSE_TYPE, BETA
    )

    # Add noise using Eb/N0 model
    received_ch, noise_ch, noise_var_ch = awgn_channel_ebn0(
        waveform_ch, SNR_DB, pulse_ch, SAMPLES_PER_BIT, Tb
    )
    signal_power = np.mean(waveform_ch**2)
    actual_snr = 10 * np.log10(signal_power / noise_var_ch) if noise_var_ch > 0 else np.inf

    print(f"  Eb/N0 target: {SNR_DB} dB")
    print(f"  Signal power: {signal_power:.4f}")
    print(f"  Noise variance: {noise_var_ch:.6f}")
    print(f"  Waveform SNR: {actual_snr:.2f} dB")

    # Eye diagram of NOISY signal (before matched filter)
    fig4a = plot_eye_diagram(
        received_ch, SAMPLES_PER_BIT, num_traces=500,
        title=f"Eye Diagram | After AWGN Channel | SNR={SNR_DB} dB | Before Matched Filter"
    )
    fig4a.savefig('outputs/block4_eye_diagram_noisy.png', dpi=150, bbox_inches='tight')
    plt.close(fig4a)

    # Show eye diagrams at different Eb/N0 levels
    snr_demo = [2, 6, 10, 20]
    fig4b, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig4b.suptitle('Eye Diagrams at Different Eb/N0 Levels (Before Matched Filter)', fontsize=13)
    for ax, snr in zip(axes.flatten(), snr_demo):
        rx, _, _ = awgn_channel_ebn0(waveform_ch, snr, pulse_ch, SAMPLES_PER_BIT, Tb)
        trace_len = 2 * SAMPLES_PER_BIT
        t_eye = np.linspace(0, 2, trace_len)
        n_tr = min(len(rx) // trace_len, 200)
        for i in range(n_tr):
            seg = rx[i*trace_len:(i+1)*trace_len]
            ax.plot(t_eye, seg, 'b-', alpha=0.1, linewidth=0.8)
        ax.set_title(f'SNR = {snr} dB')
        ax.set_xlabel('Time (normalized to Tb)')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
    fig4b.savefig('outputs/block4_eye_diagrams_snr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig4b)

    print("\n  ✅ Block 4 plots saved to outputs/")


# ==============================================================
# BLOCK 5: MATCHED FILTER
# ==============================================================
if RUN_BLOCK_5_MATCHED:
    print_block(5, "MATCHED FILTER")

    np.random.seed(42)
    n_bits_mf = 1000
    bits_mf = np.random.randint(0, 2, n_bits_mf)
    bits_mf_polar = 2 * bits_mf - 1

    t_mf, waveform_mf, pulse_mf, _ = apply_pulse_shaping(
        bits_mf_polar, Tb, SAMPLES_PER_BIT, PULSE_TYPE, BETA
    )

    received_mf, _, _ = awgn_channel_ebn0(waveform_mf, SNR_DB, pulse_mf, SAMPLES_PER_BIT, Tb)

    # Apply matched filter
    mf_output, sampled_vals = matched_filter(received_mf, pulse_mf, SAMPLES_PER_BIT)

    # Detect bits
    detected_mf = detect_bits(sampled_vals)
    ber_mf, errors_mf = calculate_ber(bits_mf, detected_mf)

    print(f"  Matched filter pulse: {PULSE_TYPE.upper()}")
    print(f"  SNR: {SNR_DB} dB")
    print(f"  Bits transmitted: {n_bits_mf}")
    print(f"  Bit errors: {errors_mf}")
    print(f"  BER: {ber_mf:.6f}")

    # Eye diagram AFTER matched filter
    fig5a = plot_eye_diagram(
        mf_output, SAMPLES_PER_BIT, num_traces=500,
        title=f"Eye Diagram | After Matched Filter | SNR={SNR_DB} dB"
    )
    fig5a.savefig('outputs/block5_eye_diagram_matched_filter.png', dpi=150, bbox_inches='tight')
    plt.close(fig5a)

    # Compare before and after matched filter
    fig5b, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig5b.suptitle(f'Eye Diagram: Before vs After Matched Filter | SNR={SNR_DB} dB', fontsize=13)
    trace_len = 2 * SAMPLES_PER_BIT
    t_eye = np.linspace(0, 2, trace_len)
    n_tr = min(len(received_mf) // trace_len, 300)
    for i in range(n_tr):
        ax1.plot(t_eye, received_mf[i*trace_len:(i+1)*trace_len], 'b-', alpha=0.08, lw=0.8)
        ax2.plot(t_eye, mf_output[i*trace_len:(i+1)*trace_len], 'g-', alpha=0.08, lw=0.8)
    for ax, lbl in zip([ax1, ax2], ['Received (Before MF)', 'After Matched Filter']):
        ax.set_title(lbl)
        ax.set_xlabel('Time (normalized to Tb)')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
        ax.grid(True, alpha=0.3)
    fig5b.savefig('outputs/block5_before_after_matched_filter.png', dpi=150, bbox_inches='tight')
    plt.close(fig5b)

    # Signal stages plot
    fig5c = plot_channel_effects(
        t_mf, waveform_mf, received_mf, mf_output,
        SAMPLES_PER_BIT, SNR_DB, title_prefix="Block 5 | "
    )
    fig5c.savefig('outputs/block5_signal_stages.png', dpi=150, bbox_inches='tight')
    plt.close(fig5c)

    print("\n  ✅ Block 5 plots saved to outputs/")


# ==============================================================
# BLOCK 6: BIT ERROR RATE (single point)
# ==============================================================
if RUN_BLOCK_6_BER:
    print_block(6, "BIT ERROR RATE")

    np.random.seed(42)
    n_bits_ber = 10_000   # 10,000 bits for a single BER estimate
    bits_ber = np.random.randint(0, 2, n_bits_ber)
    bits_ber_polar = 2 * bits_ber - 1

    t_ber, waveform_ber, pulse_ber, _ = apply_pulse_shaping(
        bits_ber_polar, Tb, SAMPLES_PER_BIT, PULSE_TYPE, BETA
    )

    received_ber, _, _ = awgn_channel_ebn0(waveform_ber, SNR_DB, pulse_ber, SAMPLES_PER_BIT, Tb)
    mf_out_ber, svals = matched_filter(received_ber, pulse_ber, SAMPLES_PER_BIT)
    det_ber = detect_bits(svals)
    ber_val, n_err = calculate_ber(bits_ber, det_ber)

    from scipy.special import erfc as _erfc
    theory_ber_single = 0.5 * _erfc(np.sqrt(10**(SNR_DB/10)))

    print(f"  Bits transmitted:     {n_bits_ber}")
    print(f"  Bit errors:           {n_err}")
    print(f"  Simulated BER:        {ber_val:.6f}")
    print(f"  Theoretical BER:      {theory_ber_single:.6f}")
    print(f"  SNR:                  {SNR_DB} dB")

    # Show first few detected vs transmitted
    n_show = 20
    min_len = min(len(bits_ber), len(det_ber), n_show)
    print(f"\n  First {n_show} bits comparison:")
    print(f"  Transmitted: {bits_ber[:min_len]}")
    print(f"  Detected:    {det_ber[:min_len]}")
    err_mask = bits_ber[:min_len] != det_ber[:min_len]
    print(f"  Errors:      {''.join(['X' if e else '.' for e in err_mask])}")

    print("\n  ✅ Block 6 complete")


# ==============================================================
# BONUS: BER WATERFALL CURVE
# ==============================================================
if RUN_BONUS_BER_CURVE:
    print_block("BONUS", "BER vs SNR WATERFALL CURVE")
    print(f"  Using {N_BITS_BER:,} bits per SNR point")
    print(f"  SNR range: {SNR_RANGE_DB} dB")
    print("  (This may take 1-2 minutes...)")

    snr_arr = np.array(SNR_RANGE_DB)
    ber_sim_arr = []

    from pulse_shaping import apply_pulse_shaping as aps

    np.random.seed(0)
    for snr_db in SNR_RANGE_DB:
        bits_w = np.random.randint(0, 2, N_BITS_BER)
        bits_w_polar = 2 * bits_w - 1

        _, wf_w, pulse_w, _ = aps(bits_w_polar, Tb, SAMPLES_PER_BIT, PULSE_TYPE, BETA)
        rx_w, _, _ = awgn_channel_ebn0(wf_w, snr_db, pulse_w, SAMPLES_PER_BIT, Tb)
        mf_w, sv_w = matched_filter(rx_w, pulse_w, SAMPLES_PER_BIT)
        det_w = detect_bits(sv_w)
        ber_w, nerr_w = calculate_ber(bits_w, det_w)
        ber_sim_arr.append(max(ber_w, 1e-7))   # avoid log(0)

        print(f"  SNR={snr_db:3d}dB | BER={ber_w:.2e} | Errors={nerr_w}/{N_BITS_BER}")

    ber_theory_arr = theoretical_ber(snr_arr)

    fig_ber = plot_ber_curve(
        snr_arr, ber_sim_arr, ber_theory_arr,
        title=f"BER vs SNR Waterfall Curve | {PULSE_TYPE.upper()} Pulse"
              + (f" β={BETA}" if PULSE_TYPE == 'rc' else "")
              + f" | {N_BITS_BER:,} bits/point"
    )
    fig_ber.savefig('outputs/bonus_ber_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close(fig_ber)

    print("\n  ✅ BER waterfall curve saved to outputs/")


# ==============================================================
# SUMMARY
# ==============================================================
print(f"\n{'='*60}")
print("  ✅ ALL BLOCKS COMPLETE!")
print(f"{'='*60}")
print("\n  Saved plots:")
for f in sorted(os.listdir('outputs')):
    print(f"    outputs/{f}")
print()

import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt


# ==============================================================
# FUNCTION: awgn_channel_ebn0 — Add noise based on Eb/N0
# ==============================================================
def awgn_channel_ebn0(signal, EbN0_dB, pulse, samples_per_bit, Tb):
    """
    Simulates AWGN channel using Eb/N0 (energy-per-bit to noise-density ratio).

    HOW WE SET NOISE:
    - Eb = energy per bit = sum(pulse²) * dt
    - EbN0_linear = 10^(EbN0_dB/10)
    - N0 = Eb / EbN0_linear  (noise one-sided PSD)
    - Noise variance per sample = N0/2 / dt
      (white noise sampled at rate 1/dt has power N0/2 per Hz × (1/dt) Hz)

    Parameters:
    -----------
    signal          : transmitted waveform
    EbN0_dB         : Eb/N0 in dB
    pulse           : pulse shape (to calculate Eb)
    samples_per_bit : samples per bit period
    Tb              : bit duration

    Returns:
    --------
    received    : noisy signal
    noise       : noise added
    noise_var   : noise variance used
    """
    dt = Tb / samples_per_bit
    Eb = np.sum(pulse ** 2)          # pulse energy (amplitude² × samples, dt cancels)
    EbN0_linear = 10 ** (EbN0_dB / 10)
    N0 = Eb / EbN0_linear             # noise one-sided PSD
    noise_var = N0 / 2               # noise variance per sample

    noise = np.random.normal(0, np.sqrt(noise_var), len(signal))
    received = signal + noise

    return received, noise, noise_var



def awgn_channel(signal, SNR_dB):
    """
    Simulates an AWGN channel by adding Gaussian noise.

    HOW IT WORKS:
    1. Calculate signal power: Ps = mean(signal²)
    2. Convert SNR from dB: SNR_linear = 10^(SNR_dB/10)
    3. Required noise power: Pn = Ps / SNR_linear
    4. Generate noise: n ~ N(0, Pn) → std = sqrt(Pn)
    5. Add noise: received = signal + noise

    Parameters:
    -----------
    signal  : numpy array — transmitted signal s(t)
    SNR_dB  : float — desired Signal-to-Noise Ratio in dB

    Returns:
    --------
    received    : noisy received signal r(t) = s(t) + n(t)
    noise       : the noise that was added
    noise_power : actual noise power (σ²)
    """
    # Signal power
    signal_power = np.mean(signal ** 2)

    # Convert SNR from dB to linear scale
    SNR_linear = 10 ** (SNR_dB / 10)

    # Required noise power
    noise_power = signal_power / SNR_linear

    # Generate Gaussian noise: mean=0, variance=noise_power
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Add noise to signal
    received = signal + noise

    return received, noise, noise_power


# ==============================================================
# FUNCTION: matched_filter — Apply matched filter at receiver
# ==============================================================
def matched_filter(received_signal, pulse, samples_per_bit):
    """
    Applies the matched filter to the received signal.

    HOW IT WORKS:
    - The matched filter h(t) = p(Tb - t) is the time-reversed pulse
    - We convolve received signal with h(t)
    - Then sample at the end of each bit period (t = Tb, 2Tb, ...)

    Parameters:
    -----------
    received_signal : numpy array — noisy received signal
    pulse           : numpy array — the pulse shape used at transmitter
    samples_per_bit : int — samples per bit period

    Returns:
    --------
    mf_output    : matched filter output (full waveform)
    sampled_vals : values sampled at bit decision instants
    """
    # Matched filter = time-reversed pulse
    h = pulse[::-1]   # flip the pulse in time

    # Convolve received signal with matched filter
    mf_full = np.convolve(received_signal, h, mode='full')

    # The matched filter has delay = (len(h)-1)//2 for symmetric pulses
    # The output at index k*spb + delay gives the decision for bit k
    delay = (len(h) - 1) // 2

    # Take only the part corresponding to original signal length
    mf_output = mf_full[delay: delay + len(received_signal)]

    # Sample at the START of each bit period (index i*samples_per_bit)
    # After the delay compensation above, the MF peak aligns here.
    n_bits = len(received_signal) // samples_per_bit
    sampled_vals = np.array([
        mf_output[i * samples_per_bit]
        for i in range(n_bits)
    ])

    return mf_output, sampled_vals


# ==============================================================
# FUNCTION: detect_bits — Make binary decisions
# ==============================================================
def detect_bits(sampled_vals, threshold=0.0):
    """
    Makes bit decisions based on matched filter output samples.

    Decision rule (for NRZ Polar):
    - sampled_val > threshold  →  detected bit = 1
    - sampled_val ≤ threshold  →  detected bit = 0

    Parameters:
    -----------
    sampled_vals : numpy array — matched filter output at sampling instants
    threshold    : float — decision threshold (default 0 for polar signaling)

    Returns:
    --------
    detected_bits : numpy array of 0s and 1s
    """
    return (sampled_vals > threshold).astype(int)


# ==============================================================
# FUNCTION: calculate_ber — Bit Error Rate
# ==============================================================
def calculate_ber(transmitted_bits, detected_bits):
    """
    Calculates the Bit Error Rate (BER).

    BER = (number of bit errors) / (total bits transmitted)

    A bit error occurs when the detected bit ≠ transmitted bit.

    Parameters:
    -----------
    transmitted_bits : numpy array of 0s and 1s (what we sent)
    detected_bits    : numpy array of 0s and 1s (what we received)

    Returns:
    --------
    ber         : Bit Error Rate (float between 0 and 1)
    num_errors  : total number of bit errors (int)
    """
    # Align lengths (in case of minor length mismatches)
    min_len = min(len(transmitted_bits), len(detected_bits))
    errors = np.sum(transmitted_bits[:min_len] != detected_bits[:min_len])
    ber = errors / min_len
    return ber, int(errors)


# ==============================================================
# FUNCTION: theoretical_ber — Q-function based BER
# ==============================================================
def theoretical_ber(SNR_dB_range, modulation='nrz_polar'):
    """
    Computes the theoretical BER for comparison with simulated BER.

    For NRZ Polar with matched filter:
    BER = Q(sqrt(2*SNR_linear)) = 0.5 * erfc(sqrt(SNR_linear))

    Note: Here SNR_linear = Eb/N0 (energy per bit to noise density ratio)

    Parameters:
    -----------
    SNR_dB_range : numpy array — SNR values in dB
    modulation   : str — modulation type (currently 'nrz_polar')

    Returns:
    --------
    ber_theory : numpy array — theoretical BER values
    """
    SNR_linear = 10 ** (np.array(SNR_dB_range) / 10)
    ber_theory = 0.5 * erfc(np.sqrt(SNR_linear))
    return ber_theory


# ==============================================================
# FUNCTION: simulate_ber_curve — Full BER vs SNR waterfall curve
# ==============================================================
def simulate_ber_curve(n_bits, Tb, samples_per_bit, pulse_type='rc',
                        beta=0.5, snr_range_db=None,
                        line_code_func=None, pulse_func=None):
    """
    Simulates the BER waterfall curve over a range of SNR values.

    HOW IT WORKS:
    -------------------------
    For each SNR value:
    1. Generate random bits
    2. Apply line coding (NRZ Polar)
    3. Apply pulse shaping
    4. Pass through AWGN channel with the given SNR
    5. Apply matched filter
    6. Sample and make decisions
    7. Count errors → compute BER
    Plot BER vs SNR on semilog scale.

    Parameters:
    -----------
    n_bits          : int — number of bits per SNR point (use ≥100,000 for accuracy)
    Tb              : float — bit duration
    samples_per_bit : int — samples per bit
    pulse_type      : str — 'rect', 'sinc', or 'rc'
    beta            : float — rolloff factor (for RC)
    snr_range_db    : list/array — SNR values to test in dB
                      Default: [0, 2, 4, 6, 8, 10, 12] dB

    Returns:
    --------
    snr_range_db : SNR values tested
    ber_simulated : simulated BER at each SNR
    ber_theory   : theoretical BER at each SNR
    """
    from pulse_shaping import apply_pulse_shaping

    if snr_range_db is None:
        snr_range_db = np.arange(0, 13, 2)

    ber_simulated = []

    print(f"\nSimulating BER curve with {n_bits} bits per SNR point...")
    print(f"Pulse: {pulse_type}" + (f" (β={beta})" if pulse_type == 'rc' else ""))
    print("-" * 50)

    for snr_db in snr_range_db:
        # Step 1: Generate random bits
        bits = np.random.randint(0, 2, n_bits)

        # Step 2: Convert to polar: 1→+1, 0→-1
        bits_polar = 2 * bits - 1

        # Step 3: Apply pulse shaping
        t, waveform, pulse, t_pulse = apply_pulse_shaping(
            bits_polar, Tb, samples_per_bit, pulse_type, beta
        )

        # Step 4: Add AWGN noise
        received, noise, noise_power = awgn_channel(waveform, snr_db)

        # Step 5: Matched filter
        mf_output, sampled_vals = matched_filter(received, pulse, samples_per_bit)

        # Step 6: Decision
        detected = detect_bits(sampled_vals)

        # Step 7: BER
        ber, n_errors = calculate_ber(bits, detected)
        ber_simulated.append(ber)

        print(f"  SNR = {snr_db:3d} dB | BER = {ber:.6f} | Errors = {n_errors}/{n_bits}")

    ber_theory = theoretical_ber(snr_range_db)

    return np.array(snr_range_db), np.array(ber_simulated), ber_theory


# ==============================================================
# FUNCTION: plot_ber_curve — Waterfall curve plot
# ==============================================================
def plot_ber_curve(snr_range_db, ber_simulated, ber_theory, title="BER vs SNR"):
    """
    Plots the BER waterfall curve (semilogarithmic scale).

    The "waterfall" shape is characteristic:
    - At low SNR: BER is high (≈0.5, essentially random guessing)
    - At high SNR: BER drops sharply (the "waterfall")
    - The gap between simulated and theoretical shows implementation loss
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Filter out BER = 0 (can't plot log(0))
    sim_valid = np.array(ber_simulated) > 0

    ax.semilogy(snr_range_db, ber_theory, 'b-o', linewidth=2,
                markersize=8, label='Theoretical BER')
    if np.any(sim_valid):
        ax.semilogy(np.array(snr_range_db)[sim_valid],
                    np.array(ber_simulated)[sim_valid],
                    'r--s', linewidth=2, markersize=8,
                    label='Simulated BER')

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-6, 1])
    ax.set_xlim([snr_range_db[0] - 0.5, snr_range_db[-1] + 0.5])

    # Add BER = 0.5 line (random guessing baseline)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='BER=0.5 (random)')

    plt.tight_layout()
    return fig


# ==============================================================
# FUNCTION: plot_channel_effects — Show signal at different stages
# ==============================================================
def plot_channel_effects(t, signal, received, mf_output,
                          samples_per_bit, SNR_dB, title_prefix=""):
    """
    Plots the signal at 3 stages: transmitted, after channel, after matched filter.
    This shows what the eye diagram looks like at each stage.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{title_prefix}Signal at Different Stages | SNR = {SNR_dB} dB",
                 fontsize=13)

    show_len = min(len(t), 30 * samples_per_bit)   # show first 30 bits

    axes[0].plot(t[:show_len], signal[:show_len], 'b-', linewidth=1.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Transmitted Signal s(t)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[:show_len], received[:show_len], 'r-', linewidth=1, alpha=0.8)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Received Signal r(t) = s(t) + noise [SNR = {SNR_dB} dB]')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t[:show_len], mf_output[:show_len], 'g-', linewidth=1.5)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.5, label='Decision threshold')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Matched Filter Output')
    axes[2].set_xlabel('Time (sec)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt


def generate_line_code(bitstream, Tb, samples_per_bit, line_code='nrz_polar', A=1.0):
    """
    Converts a bitstream into a line coded waveform.

    Parameters:
    -----------
    bitstream       : numpy array of 0s and 1s
    Tb              : float -- bit duration in seconds (= 1/bit_rate)
    samples_per_bit : int -- number of samples per bit period
    line_code       : str -- one of: 'nrz_polar', 'nrz_onoff', 'bipolar',
                                     'rz_polar', 'rz_onoff'
    A               : float -- signal amplitude (default 1.0)

    Returns:
    --------
    t        : time axis (numpy array)
    waveform : line coded waveform (numpy array)
    """
    N_bits = len(bitstream)
    dt = Tb / samples_per_bit

    t = np.arange(0, N_bits * samples_per_bit) * dt
    waveform = np.zeros(N_bits * samples_per_bit)

    last_polarity = 1

    for i, bit in enumerate(bitstream):
        start = i * samples_per_bit
        half = samples_per_bit // 2

        if line_code == 'nrz_polar':
            waveform[start: start + samples_per_bit] = A if bit == 1 else -A

        elif line_code == 'nrz_onoff':
            waveform[start: start + samples_per_bit] = A if bit == 1 else 0

        elif line_code == 'bipolar':
            if bit == 0:
                waveform[start: start + samples_per_bit] = 0
            else:
                waveform[start: start + samples_per_bit] = A * last_polarity
                last_polarity *= -1

        elif line_code == 'rz_polar':
            waveform[start: start + half] = A if bit == 1 else -A
            waveform[start + half: start + samples_per_bit] = 0

        elif line_code == 'rz_onoff':
            waveform[start: start + half] = A if bit == 1 else 0
            waveform[start + half: start + samples_per_bit] = 0

        else:
            raise ValueError(f"Unknown line code: '{line_code}'. "
                             f"Choose from: nrz_polar, nrz_onoff, bipolar, rz_polar, rz_onoff")

    return t, waveform


def plot_line_code(t, waveform, bitstream, Tb, line_code, title_prefix=""):
    """
    Plots the line coded waveform with bit markers and PSD.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle(f"{title_prefix}Line Code: {line_code.upper()} | "
                 f"Bit Rate = {1/Tb:.0f} bps | {len(bitstream)} bits", fontsize=13)

    ax1 = axes[0]
    ax1.plot(t, waveform, 'b-', linewidth=1.5)
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{line_code.upper()} Waveform')
    ax1.grid(True, alpha=0.3)

    for i in range(len(bitstream) + 1):
        ax1.axvline(x=i * Tb, color='gray', linestyle='--', alpha=0.4)

    for i, bit in enumerate(bitstream):
        ax1.text((i + 0.5) * Tb, ax1.get_ylim()[1] * 0.85, str(bit),
                 ha='center', va='center', fontsize=9, color='red', fontweight='bold')

    ax2 = axes[1]
    N = len(waveform)
    dt = t[1] - t[0]
    freqs = np.fft.fftshift(np.fft.fftfreq(N, dt))
    spectrum = np.fft.fftshift(np.fft.fft(waveform))
    psd = np.abs(spectrum) ** 2 / N

    ax2.plot(freqs, 10 * np.log10(psd + 1e-12), 'r-', linewidth=1)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (dB)')
    ax2.set_title('Power Spectral Density')
    ax2.set_xlim([-5 / Tb, 5 / Tb])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_all_line_codes(bitstream, Tb, samples_per_bit, A=1.0):
    """
    Plots all 5 line codes for the same bitstream side by side.
    """
    codes = ['nrz_polar', 'nrz_onoff', 'bipolar', 'rz_polar', 'rz_onoff']
    labels = ['NRZ Polar', 'NRZ On-Off (Unipolar)', 'Bipolar (AMI)',
              'RZ Polar', 'RZ On-Off']

    fig, axes = plt.subplots(len(codes), 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'All Line Codes | Bits: {bitstream[:20]}... | Bit Rate={1/Tb:.0f} bps',
                 fontsize=13)

    for ax, code, label in zip(axes, codes, labels):
        t, waveform = generate_line_code(bitstream, Tb, samples_per_bit, code, A)
        ax.plot(t, waveform, linewidth=1.5)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim([-1.5 * A, 1.5 * A])
        ax.grid(True, alpha=0.3)

        for i in range(min(len(bitstream), 30) + 1):
            ax.axvline(x=i * Tb, color='gray', linestyle='--', alpha=0.3)

    axes[-1].set_xlabel('Time (sec)')
    plt.tight_layout()
    return fig


def plot_eye_diagram(waveform, samples_per_bit, num_traces=None, title="Eye Diagram"):
    """
    Plots the Eye Diagram of a waveform.

    Parameters:
    -----------
    waveform        : numpy array -- the signal
    samples_per_bit : int -- samples per bit period
    num_traces      : int -- number of traces to overlay (None = all)
    title           : str -- plot title
    """
    trace_len = 2 * samples_per_bit
    n_traces = len(waveform) // trace_len
    if num_traces is not None:
        n_traces = min(n_traces, num_traces)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    t_eye = np.linspace(0, 2, trace_len)

    for i in range(n_traces):
        segment = waveform[i * trace_len: (i + 1) * trace_len]
        ax.plot(t_eye, segment, 'b-', alpha=0.1, linewidth=0.8)

    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5,
               label='Optimal sampling instant')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('Time (Normalized to Tb)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

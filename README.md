# End-to-End Digital Communication System Simulator

A modular Python-based implementation of an end-to-end baseband digital communication system. The project simulates the complete transmitter–receiver chain, including PCM generation, line coding, pulse shaping, AWGN channel modeling, matched filtering, eye-diagram analysis, and BER performance evaluation.

## Features

* Signal generation using configurable sinusoidal sources
* Uniform sampling and quantization
* Pulse Code Modulation (PCM)
* Polar NRZ, Unipolar NRZ, and Bipolar AMI line coding
* Rectangular and Raised Cosine pulse shaping
* AWGN channel simulation
* Matched filter receiver
* Eye diagram analysis
* Bit detection and BER computation
* BER vs SNR (Waterfall Curve) generation

---

## System Architecture

```text
Analog Signal
      │
      ▼
Sampling
      │
      ▼
Quantization
      │
      ▼
PCM Encoding
      │
      ▼
Line Coding
      │
      ▼
Pulse Shaping
      │
      ▼
AWGN Channel
      │
      ▼
Matched Filter
      │
      ▼
Bit Detection
      │
      ▼
BER Calculation
```

---

## Project Structure

```text
PCSproj/
│
├── sampling_quantization.py
├── line_codes.py
├── pulse_shaping.py
├── channel_matched_filter.py
├── eye_diagrams.py
├── test_code.py
├── Output_Plots/
├── plots.pdf
└── README.md
```

---

## Requirements

```bash
pip install numpy matplotlib scipy
```

---

## Running the Simulation

```bash
python test_code.py
```

All major simulation parameters such as sampling frequency, quantization levels, line coding scheme, pulse shaping parameters, and channel SNR can be modified directly in `test_code.py`.

---

## Results

The `Output_Plots` directory contains generated results for:

* Sampling and Quantization
* PCM Waveforms
* Line Coding Schemes
* Raised Cosine Pulse Shaping
* Eye Diagrams
* Matched Filter Output
* BER vs SNR Waterfall Curve

Representative outputs are also included in `plots.pdf`.

---

## Concepts Implemented

* Nyquist Sampling Theorem
* Uniform Quantization
* Pulse Code Modulation (PCM)
* Line Coding Techniques
* Pulse Shaping and Bandwidth Control
* Additive White Gaussian Noise (AWGN)
* Matched Filter Detection
* Eye Diagram Analysis
* Bit Error Rate (BER) Estimation

---

## References

1. B. P. Lathi and Zhi Ding, *Modern Digital and Analog Communication Systems*
2. Upamanyu Madhow, *Introduction to Communication Systems*

---

## Author

**Nischal Kant Singh**

Course Project – Principles of Communication Systems (PCS)

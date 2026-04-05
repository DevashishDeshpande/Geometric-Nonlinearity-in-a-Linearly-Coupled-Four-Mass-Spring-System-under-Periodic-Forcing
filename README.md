# 🧲 Geometric Nonlinearity in a Linearly Coupled Four-Mass Spring System

> A numerical study of how pure geometry — not nonlinear springs — can break the rules of linear dynamics.

**Authors:** Devashish Deshpande, Akshat Gupta  
**Institution:** Department of Physics, BITS Pilani

---

## 📌 What This Project Is About

We study a 2×2 lattice of four identical masses connected by **perfectly linear springs** to each other and to rigid walls. Despite the springs being linear, the system is **nonlinear** — because in 2D, as masses move, spring angles change, and the restoring force vector rotates. This is called **geometric nonlinearity**.

The project has two parts:
- **Part 1** — How different spatial forcing patterns at the boundaries excite different modes
- **Part 2** — Direct comparison of linear vs nonlinear dynamics: trajectories, energy, spectra, and frequency shifts

---

## 🗂️ Project Structure
```
geometric-nonlinearity-4mass-spring/
│
├── forcing_config/
│   └── forcing_config.py          # Part 1: 16 forcing cases, frequency sweeps, 3D plots
│
├── linear_vs_nonlinear/
│   └── linear_vs_nonlinear.py     # Part 2: RK4 simulation, FFT, phase portraits, freq shift
│
├── requirements.txt               # Python dependencies
└── README.md                      # You are here
```

---

## ⚙️ System Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Mass | m | 1.0 kg |
| Wall spring stiffness | K | 10.0 N/m |
| Coupling spring stiffness | Kc | 2.0 N/m |
| Wall spring rest length | L | 1.0 m |
| Coupling spring rest length | Lc | 2.0 m |
| Damping coefficient | b | 0.1 N·s/m |
| Time step | Δt | 0.005 s |
| Drive amplitude | F₀ | 0.5 N |

---

## 🎵 Natural Frequencies

The linearized system has exactly **two distinct eigenfrequencies**, each 4-fold degenerate:

| Mode Group | Formula | Value |
|------------|---------|-------|
| ω_low (symmetric) | √(K/m) | ≈ 3.162 rad/s |
| ω_high (anti-symmetric) | √((K + 2Kc)/m) | ≈ 3.742 rad/s |

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run Part 1 — Forcing Configuration
```bash
cd forcing_config
python forcing_config.py
```
Outputs are saved to `forcing_config/forcing_output_v4/`

### Run Part 2 — Linear vs Nonlinear
```bash
cd linear_vs_nonlinear
python linear_vs_nonlinear.py
```
Outputs are saved to `linear_vs_nonlinear/`

> **Note:** Part 1 uses Numba JIT compilation. The first run takes ~10–20 seconds to compile — subsequent runs are fast due to caching.

---

## 📊 Part 1 — Forcing Configuration

16 forcing cases are organized into four groups:

| Group | Description |
|-------|-------------|
| **A** | Single-mode excitation — targets ω_low or ω_high selectively |
| **B** | Phase contrast — same walls, φ=0 vs φ=π excites different modes |
| **C** | Mode mixing — diagonal and cross-axis forcing patterns |
| **D** | Reference case — all 8 walls driven in phase |

**Each case produces:**
- 3D surface: driving frequency × time × displacement
- Peak amplitude surface across all 8 DOFs
- Linear vs Nonlinear amplitude curves
- Nonlinearity residual heatmap `|NL − Lin| / Lin`

---

## 📈 Part 2 — Linear vs Nonlinear Comparison

Starting from the same initial conditions, both the full nonlinear and linearized systems are integrated with RK4. The following diagnostics are produced:

| Output File | Description |
|-------------|-------------|
| `01_displacement_vs_time.png` | All 8 DOF trajectories — nonlinear vs linear |
| `02_phase_portraits15.png` | Phase portraits for all masses, both models |
| `03_energy_vs_time.png` | Kinetic, potential, total energy with drift % |
| `04a_fft_nonlinear.png` | FFT spectrum — nonlinear, with eigenfrequency markers |
| `04b_fft_linear.png` | FFT spectrum — linear, with eigenfrequency markers |
| `05_modal_energy.png` | Modal energy transfer across all 8 normal modes |
| `06_frequency_response.png` | Frequency response curve via impedance matrix |
| `07_freq_shift.png` | Nonlinear frequency shift vs initial amplitude |
| `simulation_log.txt` | Full console output saved to file |

---

## ✅ Physics Verification (Part 1)

Before every sweep, four automated checks run and print to console:

| Check | What It Tests | Pass Condition |
|-------|--------------|----------------|
| V0 | Equilibrium stability — q=0 must not drift | Max drift < 1e-8 m |
| V1 | Free-decay FFT peaks match eigenfrequencies | Error < 5% per mode |
| V2 | Analytic bounds on ω_low and ω_high | Error < 1e-4 rad/s |
| V3 | Near-resonance amplitude vs impedance matrix | Ratio ∈ (0.5, 2.0) |

---

## 🔑 Key Physics Findings

- Geometric nonlinearity causes **amplitude-dependent frequency stiffening** — natural frequencies shift upward as oscillation amplitude grows
- Single-axis forcing induces **transverse motion** in orthogonal DOFs due to 2D spring geometry — this effect is completely absent in the linear model
- Phase of boundary forcing (φ=0 vs φ=π) completely changes **which mode is excited**, even with identical wall pairs
- All-walls-in-phase driving (Case D1) excites only ω_low modes linearly — ω_high modes engage **only through nonlinear geometric coupling**
- The system has exactly **two distinct eigenfrequencies** despite having 8 DOFs, due to the square lattice symmetry causing 4-fold degeneracy at each frequency

---

## 📦 Dependencies
```
numpy
matplotlib
scipy
numba
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📄 Reference

> Deshpande, D. & Gupta, A. — *Geometric Nonlinearity in a Linearly Coupled Four-Mass Spring System under Periodic Forcing: A Comparative Numerical Study* — Department of Physics, BITS Pilani
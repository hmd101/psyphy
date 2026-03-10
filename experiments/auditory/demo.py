import csv
import random
import sys
import termios
import time
import tty

import numpy as np
import sounddevice as sd
from scipy.stats import qmc


def getch():
    """Read a single keypress from stdin, without waiting for Enter (Unix only)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)  # put terminal into raw mode
        ch = sys.stdin.read(1)  # read one character
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# ========== Global parameters ==========
FS = 44100  # Sampling rate (Hz)
DURATION = 0.5  # Tone duration (s)
ATTACK = 0.01  # Attack time (s)
RELEASE = 0.05  # Release time (s)

# F0_MIN = 220     # Hz
# F0_MAX = 880     # Hz
F0_MIN = 400  # Hz
F0_MAX = 420  # Hz
SLOPE_MIN = -15  # dB per octave range
SLOPE_MAX = 0  # dB per octave range

F0_PERTURB_SCALE = 0.04  # ±1% perturbation in F0 for the odd one out
SLOPE_PERTURB_SCALE = 0.0  # ±1 dB/oct perturbation in slope for the odd one out

N_TRIALS = 2**7  # Total number of trials (ideally a power of 2 for Sobol sampling)
ISI = 0.1
LOGFILE = "tone_discrimination_log.csv"

# ========== Stimulus generation ==========


def adsr_envelope(n_samples, fs, attack, release):
    t = np.arange(n_samples) / fs
    env = np.ones_like(t)

    a_samp = int(attack * fs)
    if a_samp > 0:
        env[:a_samp] = np.linspace(0, 1, a_samp)

    r_samp = int(release * fs)
    if r_samp > 0:
        env[-r_samp:] = np.linspace(1, 0, r_samp)

    return env


def harmonic_complex(F0, fs, duration, slope_db_per_oct, n_harmonics=20):
    t = np.arange(int(duration * fs)) / fs
    sig = np.zeros_like(t)

    for k in range(1, n_harmonics + 1):
        freq = k * F0
        if freq >= fs / 2:
            break
        octaves = np.log2(k)
        amp_db = slope_db_per_oct * octaves
        amp = 10 ** (amp_db / 20.0)
        sig += amp * np.sin(2 * np.pi * freq * t)

    sig /= np.max(np.abs(sig)) + 1e-12
    env = adsr_envelope(len(sig), fs, ATTACK, RELEASE)
    sig *= env
    return sig.astype(np.float32)


def play_tone(signal, fs):
    sd.play(signal, fs, blocking=True)


# ========== Trial logic ==========


def random_perturbation(F0_A, slope_A):
    """Apply a random perturbation to F0 and slope."""
    F0_B = F0_A * 10 ** random.uniform(
        np.log10(1 - F0_PERTURB_SCALE), np.log10(1 + F0_PERTURB_SCALE)
    )
    slope_B = slope_A + random.uniform(-SLOPE_PERTURB_SCALE, SLOPE_PERTURB_SCALE)
    return F0_B, slope_B


def sobol_2d_samples():
    # Sobol engine in 2D
    engine = qmc.Sobol(d=2, scramble=True)

    # For best properties, n_points should ideally be a power of 2;
    # if not, scipy will warn but still generate points.
    # You can also use random_base2(m) with n_points = 2**m.
    sample = engine.random(N_TRIALS)  # shape (n_points, 2) in [0,1)

    x = sample[:, 0]  # for F0
    y = sample[:, 1]  # for slope

    # Map x to log-uniform F0
    logF_min = np.log10(F0_MIN)
    logF_max = np.log10(F0_MAX)
    logF = logF_min + x * (logF_max - logF_min)
    F0 = 10**logF

    # Map y to linear slope in dB/oct
    slope = SLOPE_MIN + y * (SLOPE_MAX - SLOPE_MIN)

    return F0, slope


# ========== Main loop ==========


def run_experiment():

    F0s, slopes = sobol_2d_samples()

    with open(LOGFILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial_index",
                "true_odd_one_out",
                "user_response",
                "correct",
                "F0_A",
                "slope_A",
                "F0_B",
                "slope_B",
                "rt_sec",
            ]
        )

        num_correct = 0

        for trial_idx in range(1, N_TRIALS + 1):
            F0_A = F0s[trial_idx - 1]
            slope_A = slopes[trial_idx - 1]
            F0_B, slope_B = random_perturbation(F0_A, slope_A)

            tone_A = harmonic_complex(F0_A, FS, DURATION, slope_A)
            tone_B = harmonic_complex(F0_B, FS, DURATION, slope_B)

            tones = [tone_A, tone_A, tone_B]
            perm = np.random.permutation(3).tolist()
            true_odd_one_out = perm.index(2)
            print("perm:", perm, "true odd one out position:", true_odd_one_out + 1)

            print(f"\nTrial {trial_idx}/{N_TRIALS}")
            play_tone(tones[perm[0]], FS)
            time.sleep(ISI)
            play_tone(tones[perm[1]], FS)
            time.sleep(ISI)
            play_tone(tones[perm[2]], FS)

            print("Odd one out? [1/2/3] ", end="", flush=True)
            t0 = time.time()
            resp_key = getch().strip().lower()
            rt = time.time() - t0

            if resp_key in ["1", "2", "3"]:
                resp_odd = int(resp_key) - 1  # subtract one to get 0-based index
                correct = resp_odd == true_odd_one_out
                num_correct += int(correct)
            else:
                resp_odd = None
                correct = None

            writer.writerow(
                [
                    trial_idx,
                    true_odd_one_out,
                    resp_odd,
                    correct,
                    F0_A,
                    slope_A,
                    F0_B,
                    slope_B,
                    f"{rt:.3f}",
                ]
            )
            f.flush()

            print("Current accuracy: {:.1f}%".format(num_correct / trial_idx * 100))

    print(f"\nExperiment finished. Data saved to {LOGFILE}")


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # F0s, slopes = sobol_2d_samples()
    # plt.scatter(F0s, slopes)
    # plt.xscale("log")
    # plt.xlabel("F0 (Hz)")
    # plt.ylabel("Slope (dB/oct)")
    # plt.title("Sampled (F0, slope) pairs using 2D Fibonacci lattice")
    # plt.show()

    run_experiment()

import numpy as np
import mynetworks.util as util

# TIME (measured in units of time_constant_rates_evoked=10ms)
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

# PARAMETERS
LEARNING_RATE = 0.14
STDP_INTEGRAL = -(4 / 30)
STDP_POTENTIATION_TIME_CONSTANT = 2.5
STDP_POTENTIATION_AMPLITUDE = 0.08
STDP_DEPRESSION_TIME_CONSTANT = 5.0

# DERIVED
stdp_depression_amplitude = util.get_remaining_amplitude_for_symmetric_stdp(
    STDP_INTEGRAL,
    STDP_DEPRESSION_TIME_CONSTANT,
    STDP_POTENTIATION_TIME_CONSTANT,
    STDP_POTENTIATION_AMPLITUDE,
)


def stdp(delta_t):
    delta_t_abs = np.absolute(delta_t)

    potentiation = STDP_POTENTIATION_AMPLITUDE * np.exp(
        -delta_t_abs / STDP_POTENTIATION_TIME_CONSTANT
    )
    depression = stdp_depression_amplitude * np.exp(
        -delta_t_abs / STDP_DEPRESSION_TIME_CONSTANT
    )

    return potentiation + depression

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{newtxtext}"
            r"\usepackage{newtxmath}"
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{siunitx}"
        ),
    }
)

t = np.linspace(-25, 25, 501)

fig, ax = plt.subplots(figsize=(3.2, 2.4), constrained_layout=True)

ax.set_xlabel(r"\((t_i - t_{\kern-.1em j}) / \unit{\milli\second}\)")
ax.set_xlim(-250, 250)

ax.set_ylabel(r"\(\Delta W_{i{\kern-.1em j}}\)")
ax.set_ylim(-0.01, 0.028)
ax.set_yticks([-0.01, 0.00, 0.01, 0.02])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axhline(color="black", linewidth=0.8, linestyle=":")
ax.plot(t / millisecond, stdp(t))

plt.savefig("../figs/stdp.pdf")

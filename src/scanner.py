import os
import itertools
import numpy as np

job_name = "which_tau_add"
script = f"{job_name}.py"
job_name += "_shuffled"
directory = "scan_shuffled"
# time units
millisecond = 1 / 10
second = 100
minute = 60 * second
hour = 60 * minute
day = 24 * hour

learning_rate = 0.14
# parameters ranges to scan
seed_rng_list = 2000 + np.arange(36)
time_constant_normalization_list = [
    1 * minute,
    10 * minute,
    1 * hour,
    4 * hour,
    12 * hour,
    1 * day,
    3 * day,
    10 * day,
]

parameters = itertools.product(
    seed_rng_list,
    # time_constant_normalization_list,
)

for (
    seed_rng,
    # time_constant_normalization,
) in parameters:
    descriptor = f"{job_name}_tau=None_seed={seed_rng}"
    # descriptor = f"{job_name}_tau={time_constant_normalization:08.0f}-seed={seed_rng}"
    os.system(
        f"sbatch --job-name={descriptor}.job "
        f"--output=../{directory}/logs/{descriptor}.out "
        f"--error=../{directory}/logs/{descriptor}.err "
        # f"{script} {learning_rate} {time_constant_normalization} {seed_rng}"
        f"{script} {learning_rate} {seed_rng}"
    )

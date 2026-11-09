"""Constants, search grids, scenarios and run sizes."""

from dataclasses import dataclass

import numpy as np

# Experiment 1
R = 1.0            # clean observation noise variance
EPS = 0.1          # long-run fraction of readings that are outliers
LAM = 100.0        # outliers have variance LAM * R, i.e. standard deviation 10
JUMP = 5.0         # size of the real change in the state
M0, P0 = 0.0, 1.0  # prior on the initial state
NEAR = 1.0         # the jump counts as followed once the average error is below NEAR (80%)
AFTER = 50         # steps after the jump used for the post-jump RMSE
C_PRIOR = 2.0      # scale fixed in advance: a 1-sd residual keeps w = 0.89
K_PRIOR = 5        # window fixed in advance for the input-dependent noise
N_BOOT = 2000      # bootstrap resamples of the test sequences, for the jump delay
GRID_DX = 0.025    # grid spacing of the Bayes filter

R_GRID = np.logspace(-2, 3, 101)  # candidate R values from 0.01 to 1000, 20 per decade
C_GRID = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 5, 6, 8, 10, 15,
                   np.inf])       # inf switches the down-weighting off
K_GRID = (3, 5, 9)                # window lengths tried for the input-dependent noise

Z90 = 1.6448536269514722  # 95% point of N(0, 1)


@dataclass(frozen=True)
class Outliers:
    """How readings are corrupted: long-run rate eps, variance multiplier lam, and persistence.

    stay = None gives independent outliers; a number p gives P(outlier | outlier at the
    previous step) = p.
    """
    eps: float
    lam: float
    stay: float | None = None


@dataclass(frozen=True)
class Scenario:
    """A setting of experiment 1 and the sweep."""
    name: str          # heading in the text output and the results
    tex: str           # row label of the sweep tables
    Q: float           # process noise variance
    train: Outliers    # outliers in the training and validation sequences
    test: Outliers     # outliers in the test sequences


SCENARIOS = [
    Scenario("main: 10% outliers, sd 10", r"10\% outliers, sd 10 (main)",
             0.1, Outliers(0.1, 100.0), Outliers(0.1, 100.0)),
    Scenario("clean data", r"none (clean data)",
             0.1, Outliers(0.0, 100.0), Outliers(0.0, 100.0)),
    Scenario("rare outliers: 5%, sd 10", r"5\%, sd 10",
             0.1, Outliers(0.05, 100.0), Outliers(0.05, 100.0)),
    Scenario("frequent outliers: 20%, sd 10", r"20\%, sd 10",
             0.1, Outliers(0.2, 100.0), Outliers(0.2, 100.0)),
    Scenario("frequent mild outliers: 20%, sd 3", r"20\%, sd 3",
             0.1, Outliers(0.2, 9.0), Outliers(0.2, 9.0)),
    Scenario("huge outliers: 10%, sd 32", r"10\%, sd 32",
             0.1, Outliers(0.1, 1000.0), Outliers(0.1, 1000.0)),
    Scenario("outliers at test time only: 10%, sd 10", r"10\%, sd 10, test time only",
             0.1, Outliers(0.0, 100.0), Outliers(0.1, 100.0)),
    Scenario("outlier bursts: 10%, sd 10, mean run 5", r"10\%, sd 10, in bursts",
             0.1, Outliers(0.1, 100.0, 0.8), Outliers(0.1, 100.0, 0.8)),
    Scenario("fast-moving state: Q = 1, 10% outliers, sd 10", r"10\%, sd 10, fast state",
             1.0, Outliers(0.1, 100.0), Outliers(0.1, 100.0)),
]

# Experiment 2
SIGMA2 = 1.0         # stationary variance of the decaying state
R2 = 0.25            # clean observation noise variance
C2 = 2.0             # robust scale
TOLS = (1e-2, 1e-8)  # passes are counted until every mean is within TOL posterior sd
REGIMES = [          # (label, median a_t, cap on a_t, seed id); None = random walk, a_t = 1
    ("a_t <= 0.6", 0.3, 0.6, 105),
    ("a_t ~ 0.5", 0.5, None, 100),
    ("a_t ~ 0.9", 0.9, None, 101),
    ("a_t ~ 0.99", 0.99, None, 102),
    ("a_t ~ 0.999", 0.999, None, 103),
    ("random walk", None, None, 104),
]


@dataclass(frozen=True)
class RunConfig:
    """Sizes of a run."""
    T: int               # steps per sequence in experiment 1
    jump_t: int          # the state jumps after this many steps
    n_train: int         # training, validation and test sequences
    n_val: int
    n_test: int
    n_stability: int     # refits on fresh data, for the tuning-stability check
    T2_list: tuple       # sequence lengths in experiment 2
    T2_mid: int          # the length at which experiment 2 also runs c = 1
    n2: int              # sequences per regime and length in experiment 2
    maxpass: int         # passes counted at most
    scenarios: tuple     # the sweep's settings, as indices into SCENARIOS


FULL = RunConfig(T=1000, jump_t=500, n_train=100, n_val=100, n_test=200, n_stability=5,
                 T2_list=(100, 1000, 10000), T2_mid=1000, n2=10, maxpass=400,
                 scenarios=tuple(range(len(SCENARIOS))))
QUICK = RunConfig(T=300, jump_t=150, n_train=10, n_val=10, n_test=20, n_stability=2,
                  T2_list=(100, 300), T2_mid=300, n2=3, maxpass=100,
                  scenarios=(0, 1, 6, 7, 8))

# The filters of experiment 1; the labels are the keys of the results
ONE_TRUE = "Kalman, R = 1"
ONE_ML, ONE_FC = "Kalman, one R (max likelihood)", "Kalman, one R (forecast CRPS)"
INDEP, INDEP_FIX = "Kalman, input-dependent R_t", f"Kalman, input-dependent R_t, c = {C_PRIOR:g} fixed"
INDEP_PAST = "Kalman, input-dependent R_t (past only)"
ROB, ROB_FIX, BOTH = "Robust, tuned", f"Robust, c = {C_PRIOR:g} fixed", "Robust + input-dependent R_t"
BOTH_PAST = "Robust + input-dependent R_t (past only)"
BAYES = "Bayes filter (reference)"
FILTERS = [ONE_TRUE, ONE_ML, ONE_FC, INDEP, INDEP_FIX, INDEP_PAST, ROB, ROB_FIX, BOTH, BOTH_PAST, BAYES]
PAIRS = [(ROB, ONE_ML), (ROB, ONE_FC), (ROB, INDEP), (BOTH, INDEP), (BOTH_PAST, INDEP_PAST),
         (ROB_FIX, INDEP_FIX), (ROB, BAYES)]  # the paired comparisons, first minus second

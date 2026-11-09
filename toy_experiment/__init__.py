"""Toy experiments for the Warwick MSc by Research proposal (Section 4). run.py runs them.

Experiment 1 (RQ1 in miniature)
    The Gaussian random walk of Sarkka (2013), Example 4.1, here with Q = 0.1:
        x_t = x_{t-1} + q_t,   q_t ~ N(0, Q),   Q = 0.1
        y_t = x_t + r_t
    Most r_t ~ N(0, R) with R = 1, but 10% of the readings are outliers, r_t ~ N(0, 100 R).
    In the test sequences the true state is 5 higher from step 501 on. The filters (all
    know Q and the prior on x_0):
      * Kalman filter with R = 1, the clean noise level, treating every reading as clean.
      * Kalman filter with one R for every step: learned by maximum likelihood, or chosen
        by the forecast criterion below.
      * Kalman filter with input-dependent noise R_t = R + d_t^2 / c^2, where d_t is the
        distance from y_t to the median of the previous k readings (the prior mean stands
        in for readings before the first). It stands in for a network that sets the noise
        from the inputs, y_t included. Tuned (R, c, k with k in {3, 5, 9}), and with
        c = 2, k = 5 fixed.
      * The same with noise from past readings only, R_t = R + d_{t-1}^2 / c^2 (R_1 = R),
        which uses y_1, ..., y_{t-1}: a network that must set the noise before it sees
        y_t, as a proper forecast density of y_t requires. Tuned (R, c, k).
      * Robust filter: the Kalman update with R replaced by R / w_t^2, using the weight of
        eq. (3). Tuned (R, c), and with c = 2 fixed.
      * Robust filter on top of the input-dependent noise, a stand-in for the model RQ1 asks
        about: R_t as above, and the weight of eq. (3) measured against P_{t|t-1} + R_t.
        R, the robust c and the noise's c and k are tuned together, so the search includes
        the input-dependent filter (robust c = inf) and the robust filter (noise c = inf).
        Once with each of the two kinds of input-dependent noise.
      * The Bayes filter for the outlier model, computed on a fine grid. It knows how the
        outliers are generated but, like every other filter, not about the jump, so it has
        the lowest expected error before the jump and is only a reference after it.
    Every free parameter is fitted on training and validation sequences that have no jump,
    using observations only: the one-R Kalman filter's R maximises the likelihood on the
    training sequences, and everything else minimises the CRPS of the one-step forecast
    N(m_{t|t-1}, P_{t|t-1} + R) of y_t on the validation sequences. The forecast uses the
    base R for every filter, the past-only ones included, so that all are scored alike: with
    an R_t that depends on y_t, N(y_t; m, P + R_t) is not a density in y_t (its integral
    over y_t is infinite), so it cannot score a forecast.
    Accuracy and calibration are measured before the jump, and the response to the jump
    separately.

Experiment 2 (RQ2 in miniature)
    The same kind of outliers, with input-dependent decay as in eq. (1):
        x_t = a_t x_{t-1} + q_t,   a_t = exp(-Delta_t),   q_t ~ N(0, (1 - a_t^2) SIGMA2)
    Delta_t follows a slowly varying input. It never depends on the weights, as in the
    proposal, where the network computes it from the observations. The robust filter is
    computed the way Section 5.2 proposes: start with every weight at 1, run a Kalman
    filter with the weights frozen, recompute every weight from that run's predictions,
    and repeat. One frozen-weight run is what one parallel scan computes (Sarkka and
    Garcia-Fernandez, 2021). Here it is run step by step, which gives the same numbers up
    to round-off.
    The experiment counts the passes needed to match the one-step-at-a-time robust filter.

Sweep
    Experiment 1 under eight other settings: clean data, rarer, more frequent, milder and
    larger outliers, outliers that only appear at test time, outliers in bursts, and a
    faster-moving state with the main setting's outliers.
"""

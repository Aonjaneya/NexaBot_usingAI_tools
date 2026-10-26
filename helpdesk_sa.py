"""
=====================================================================
 IT Helpdesk Triage Bot - Question Ordering via Simulated Annealing
=====================================================================
Goal: find the order of 8 Wi-Fi/Network troubleshooting steps that
minimises the EXPECTED cost (time + user frustration) of resolving
a ticket, while respecting prerequisite rules.

Libraries: standard library + numpy + matplotlib only.

---------------------------------------------------------------------
 MATHEMATICAL MODEL
---------------------------------------------------------------------
Let a permutation be  pi = (pi_1, ..., pi_N)  with N = 8.
Step i has time t_i (s), resolve-probability p_i, frustration f_i (1-10).

The user only reaches position k if positions 1..k-1 all FAILED:

    R_k = prod_{j=1}^{k-1} (1 - p_{pi_j}),          R_1 = 1

Expected Time to Resolution (ETR) and Expected Frustration (EF):

    ETR(pi) = sum_{k=1}^{N} R_k * t_{pi_k}
    EF(pi)  = sum_{k=1}^{N} R_k * f_{pi_k}

Prerequisite violations V(pi) = number of (A -> B) rules where
step B is placed BEFORE step A in the sequence.

Final cost (everything in "seconds-equivalent" so it can be summed):

    E(pi) = ETR(pi)  +  w_f * EF(pi)  +  LAMBDA * V(pi)

  * w_f    : seconds of user patience we "charge" per frustration point
  * LAMBDA : heavy soft-penalty per violated prerequisite (so large that
             SA essentially never accepts an invalid order once T is
             below a few hundred, yet the landscape stays connected
             so the search can still walk *through* infeasible orders
             early on).

Note: the overall chance that SOME step resolves the issue,
1 - prod(1 - p_i), is the same for every order - ordering only
changes HOW MUCH effort the user burns before getting there.
=====================================================================
"""

import math
import random
import itertools

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. PROBLEM DATA: 8 Wi-Fi / network troubleshooting steps
# ---------------------------------------------------------------------
STEPS = {
    # name                          : time (s), P(resolve), frustration (1-10)
    "Check Wi-Fi Switch":            dict(time_cost=15,  success_prob=0.12, frustration_score=1),
    "Forget & Reconnect Network":    dict(time_cost=45,  success_prob=0.15, frustration_score=3),
    "Restart Router/Modem":          dict(time_cost=150, success_prob=0.20, frustration_score=6),
    "Check VPN Split-Tunnel":        dict(time_cost=60,  success_prob=0.08, frustration_score=5),
    "Flush DNS Cache":               dict(time_cost=30,  success_prob=0.10, frustration_score=3),
    "Reinstall Network Driver":      dict(time_cost=240, success_prob=0.25, frustration_score=9),
    "Run Network Troubleshooter":    dict(time_cost=90,  success_prob=0.18, frustration_score=4),
    "Check Proxy Settings":          dict(time_cost=50,  success_prob=0.06, frustration_score=4),
}

# Prerequisite rules as (must_come_first, must_come_later)
PREREQUISITES = [
    ("Check Wi-Fi Switch",     "Reinstall Network Driver"),
    ("Check VPN Split-Tunnel", "Flush DNS Cache"),
]

W_FRUSTRATION = 10.0     # 1 frustration point == 10 seconds-equivalent
LAMBDA_PENALTY = 5000.0  # per violated prerequisite

# The "human-guessed" runbook order (valid, but unoptimised):
# someone put the big "fix-it-all" steps up front.
NAIVE_ORDER = [
    "Check Wi-Fi Switch",
    "Reinstall Network Driver",
    "Restart Router/Modem",
    "Check VPN Split-Tunnel",
    "Flush DNS Cache",
    "Forget & Reconnect Network",
    "Run Network Troubleshooter",
    "Check Proxy Settings",
]


# ---------------------------------------------------------------------
# 2. COST FUNCTION
# ---------------------------------------------------------------------
def breakdown(sequence):
    """Return (ETR, EF, n_violations) for a sequence of step names."""
    reach = 1.0            # R_k: probability the user is still in the flow
    etr = 0.0
    ef = 0.0
    for name in sequence:
        s = STEPS[name]
        etr += reach * s["time_cost"]          # time only paid if we get here
        ef += reach * s["frustration_score"]   # frustration likewise
        reach *= (1.0 - s["success_prob"])     # chance this step did NOT fix it

    position = {name: idx for idx, name in enumerate(sequence)}
    violations = sum(1 for first, later in PREREQUISITES
                     if position[later] < position[first])
    return etr, ef, violations


def calculate_cost(sequence):
    """E(pi) = ETR + w_f * EF + LAMBDA * (#violated prerequisites)."""
    etr, ef, violations = breakdown(sequence)
    return etr + W_FRUSTRATION * ef + LAMBDA_PENALTY * violations


# ---------------------------------------------------------------------
# 3. SIMULATED ANNEALING
# ---------------------------------------------------------------------
def get_neighbor(sequence, rng):
    """
    NEIGHBOR GENERATION OPERATOR
    ----------------------------
    Pick two distinct positions i < j and modify the sequence locally:
      * 50%: SWAP        - exchange the steps at positions i and j
      * 50%: 2-OPT FLIP  - reverse the whole segment between i and j
    Both keep the result a valid permutation (no step lost/duplicated)
    and make a SMALL change, so neighbouring solutions have similar cost -
    that "smoothness" is what lets annealing work.
    """
    new_seq = sequence[:]                      # never mutate the current state
    i, j = sorted(rng.sample(range(len(new_seq)), 2))
    if rng.random() < 0.5:
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]          # swap
    else:
        new_seq[i:j + 1] = reversed(new_seq[i:j + 1])            # 2-opt
    return new_seq


def simulated_annealing(initial_seq,
                        t_start=200.0,      # initial temperature (cost units)
                        t_end=0.01,         # stop when system is "frozen"
                        alpha=0.99,         # geometric cooling factor
                        iters_per_temp=10,  # moves tried at each temperature
                        seed=42):
    """
    Minimise calculate_cost() with Simulated Annealing.

    COOLING SCHEDULE (geometric):   T_new = T * alpha     (0 < alpha < 1)
      High T -> accepts many worse moves (explores, escapes local minima)
      Low  T -> accepts almost only improvements (exploits / converges)

    METROPOLIS ACCEPTANCE CRITERION:
      delta = cost(candidate) - cost(current)
      * delta <= 0 : always accept (it's better or equal)
      * delta  > 0 : accept with probability  exp(-delta / T)
                     e.g. delta=30, T=100 -> 74%;  delta=30, T=5 -> 0.25%
    """
    rng = random.Random(seed)

    current = initial_seq[:]
    current_cost = calculate_cost(current)
    best, best_cost = current[:], current_cost

    T = t_start
    history_current, history_best, history_temp = [], [], []
    uphill_accepted = 0

    while T > t_end:
        for _ in range(iters_per_temp):
            candidate = get_neighbor(current, rng)               # 1) propose
            delta = calculate_cost(candidate) - current_cost     # 2) evaluate

            # 3) Metropolis criterion
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                if delta > 0:
                    uphill_accepted += 1                         # escaped uphill!
                current = candidate
                current_cost += delta

                if current_cost < best_cost:                     # track the best ever seen
                    best, best_cost = current[:], current_cost

            history_current.append(current_cost)
            history_best.append(best_cost)
            history_temp.append(T)

        T *= alpha                                               # 4) cool down

    return best, best_cost, dict(current=history_current, best=history_best,
                                 temp=history_temp, uphill=uphill_accepted)


# ---------------------------------------------------------------------
# 4. HELPERS: baselines and verification
# ---------------------------------------------------------------------
def random_valid_order(rng):
    """Rejection-sample a random permutation that satisfies prerequisites."""
    names = list(STEPS)
    while True:
        rng.shuffle(names)
        if breakdown(names)[2] == 0:
            return names[:]


def brute_force_optimum():
    """8! = 40,320 permutations - cheap enough to verify SA found the true optimum."""
    best = min(itertools.permutations(STEPS), key=lambda p: calculate_cost(list(p)))
    return list(best), calculate_cost(list(best))


def pct_reduction(before, after):
    return 100.0 * (before - after) / before


def print_order(title, seq):
    etr, ef, v = breakdown(seq)
    print(f"\n{title}")
    for k, name in enumerate(seq, 1):
        print(f"   {k}. {name}")
    print(f"   -> ETR = {etr:7.1f} s | Exp. frustration = {ef:5.2f} "
          f"| Violations = {v} | Cost E = {calculate_cost(seq):7.1f}")


# ---------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------
def main():
    rng = random.Random(7)

    # ---- Baselines --------------------------------------------------
    random_orders = [random_valid_order(rng) for _ in range(2000)]
    random_costs = [calculate_cost(o) for o in random_orders]
    random_etr = np.mean([breakdown(o)[0] for o in random_orders])
    random_ef = np.mean([breakdown(o)[1] for o in random_orders])
    random_mean_cost = float(np.mean(random_costs))

    naive_cost = calculate_cost(NAIVE_ORDER)

    # ---- Run SA from a RANDOM (valid) starting order -----------------
    # (SA also copes with invalid starts thanks to the penalty term, but a
    #  5,000+ cost spike at iteration 0 would squash the convergence plot.)
    start = random_valid_order(random.Random(3))
    best_seq, best_cost, hist = simulated_annealing(start)

    # ---- Verify against exhaustive search ---------------------------
    opt_seq, opt_cost = brute_force_optimum()

    # ---- Console report ---------------------------------------------
    print("=" * 66)
    print(" HELPDESK TRIAGE BOT - QUESTION ORDER OPTIMISATION (SA)")
    print("=" * 66)
    print_order("BEFORE: Naive (human-guessed) order", NAIVE_ORDER)
    print_order("AFTER : Simulated-Annealing optimised order", best_seq)

    n_etr, n_ef, _ = breakdown(NAIVE_ORDER)
    s_etr, s_ef, _ = breakdown(best_seq)

    print("\n" + "-" * 66)
    print(f"{'Metric':<28}{'Naive':>10}{'SA':>10}{'Reduction':>14}")
    print("-" * 66)
    print(f"{'Expected Time (s)':<28}{n_etr:>10.1f}{s_etr:>10.1f}{pct_reduction(n_etr, s_etr):>13.1f}%")
    print(f"{'Expected Frustration':<28}{n_ef:>10.2f}{s_ef:>10.2f}{pct_reduction(n_ef, s_ef):>13.1f}%")
    print(f"{'Total Cost E':<28}{naive_cost:>10.1f}{best_cost:>10.1f}{pct_reduction(naive_cost, best_cost):>13.1f}%")
    print("-" * 66)
    print(f"Vs. average random valid order: ETR {random_etr:.1f}s -> {s_etr:.1f}s "
          f"({pct_reduction(random_etr, s_etr):.1f}% lower), "
          f"frustration {random_ef:.2f} -> {s_ef:.2f} ({pct_reduction(random_ef, s_ef):.1f}% lower)")
    print(f"\nSA iterations: {len(hist['current'])} | uphill moves accepted: {hist['uphill']}")
    print(f"Brute-force global optimum cost: {opt_cost:.2f} | SA best cost: {best_cost:.2f} "
          f"-> {'SA FOUND THE GLOBAL OPTIMUM' if math.isclose(opt_cost, best_cost) else 'SA is near-optimal'}")

    # ---- Figure -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: convergence. Current-cost line shows uphill escapes early.
    iters = np.arange(len(hist["current"]))
    ax1.plot(iters, hist["current"], color="tab:blue", alpha=0.45, lw=0.8,
             label="Current solution cost (accepts uphill moves)")
    ax1.plot(iters, hist["best"], color="tab:red", lw=2, label="Best cost so far")
    ax1.axhline(opt_cost, color="green", ls="--", lw=1, label="Brute-force optimum")
    ax1.set_ylim(opt_cost * 0.97, max(hist["current"]) * 1.03)
    ax1.set_xscale("symlog", linthresh=500)  # zoom into the hot early phase
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Expected Resolution Cost E(S)")
    ax1.set_title("SA Convergence: explore (hot) -> exploit (cold)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: horizontal bars
    labels = ["SA-Optimized Order", "Naive Order", "Random Order (avg)"]
    values = [best_cost, naive_cost, random_mean_cost]
    bars = ax2.barh(labels, values, color=["tab:green", "tab:orange", "tab:gray"])
    for b, v in zip(bars, values):
        ax2.text(v + max(values) * 0.01, b.get_y() + b.get_height() / 2,
                 f"{v:.0f}", va="center", fontweight="bold")
    ax2.set_xlim(0, max(values) * 1.15)
    ax2.set_xlabel("Expected Cost E(S)  (seconds-equivalent)")
    ax2.set_title("Expected Cost by Question Order")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("IT Helpdesk Triage Bot - Simulated Annealing Optimisation", fontweight="bold")
    fig.tight_layout()
    fig.savefig("sa_helpdesk_results.png", dpi=150)
    print("\nFigure saved to sa_helpdesk_results.png")
    plt.show()


if __name__ == "__main__":
    main()

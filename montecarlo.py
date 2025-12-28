import random
import math
import statistics
import matplotlib.pyplot as plt

# Intrusion graph
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ----------------------------
# CONFIG
# ----------------------------
SEED = 42
NUM_SIMULATIONS = 5000
MAX_ATTEMPTS = 20

# Threat model: attacker may learn/subvert delimiting if prompt/tokens are inferable/leaked
ATTACKER_KNOWS_PROMPT = True
LEARN_LAMBDA = 0.35  # higher => faster leak/learning

# Datamarking "token guessing" per Spotlighting adversary considerations (toy)
N_CHARS = 62
K_GRAM = 3
COLLISION_PROB = 1.0 / (N_CHARS ** K_GRAM)  # ~4.2e-6 for (62,3)

# Per-step leak hazard chosen so that P(leak by attempt t) ≈ 1-exp(-LEARN_LAMBDA*(t-1))
# (i.e., memoryless exponential in discrete time)
P_LEAK_PER_STEP = 1.0 - math.exp(-LEARN_LAMBDA)

# Paper replication targets (Summarization)
PAPER_ASR_SUMMARIZATION = {
    "None": 0.60,
    "InstructionsOnly": 0.60,  # keep equal to baseline to avoid inventing a number
    "Delimiting": 0.30,
    "Datamarking": 0.031,
    "Encoding": 0.0,
}

DEFENSES = ["None", "InstructionsOnly", "Delimiting", "Datamarking", "Encoding"]


# ----------------------------
# Helpers
# ----------------------------
def asr_static(defense: str) -> float:
    return PAPER_ASR_SUMMARIZATION[defense]


# ----------------------------
# Simulation
# ----------------------------
def run_static_test(defense: str, n: int) -> float:
    """Static benchmark: 1 attempt per episode (paper-like single-shot ASR)."""
    p = asr_static(defense)
    wins = 0
    for _ in range(n):
        wins += (random.random() < p)
    return wins / n


def run_dynamic_twin(defense: str, n: int, K: int):
    """
    Dynamic Security Twin:
    - Up to K attempts per episode.
    - Compromised if ANY attempt succeeds (session risk).
    - Delimiting: stochastic 'leak/learning' event => after leak, p becomes baseline.
    - Datamarking: rare token-guess event => once guessed in-session, p becomes baseline.
    - Encoding: fixed at 0 (paper-like for summarization).
    Returns: compromise_rate, steps_to_compromise, hacked_by_step, traces
    """
    compromised = 0
    steps_to_compromise = []
    hacked_by_step = [0] * (K + 1)
    traces = []

    for _ in range(n):
        # Twin state variables (toy, but now meaningful)
        delim_leaked = False
        token_guessed = False

        state = (0, delim_leaked, token_guessed)  # (attempt_index, delim_leaked, token_guessed)
        trace = []

        success = False

        for t in range(1, K + 1):
            # --- Update attacker knowledge / rare events (before attempt) ---
            if defense == "Delimiting" and ATTACKER_KNOWS_PROMPT and not delim_leaked and t >= 2:
                # leak/learning happens with a per-step hazard; once leaked it persists
                if random.random() < P_LEAK_PER_STEP:
                    delim_leaked = True

            if defense == "Datamarking" and not token_guessed:
                # rare token-guess; once guessed it persists for the session
                if random.random() < COLLISION_PROB:
                    token_guessed = True

            # --- Determine effective per-attempt success probability ---
            p_attack = asr_static(defense)

            # If a defense collapses, revert to baseline
            if defense == "Delimiting" and delim_leaked:
                p_attack = asr_static("None")

            if defense == "Datamarking" and token_guessed:
                p_attack = asr_static("None")

            # Encoding fixed at 0 for summarization (paper-like)
            if defense == "Encoding":
                p_attack = 0.0

            # --- Monte Carlo attempt ---
            outcome = (random.random() < p_attack)

            # Intrusion-graph next state:
            # - on success: go to absorbing COMPROMISED node
            # - on fail: go to next time state
            if outcome:
                next_state = ("COMPROMISED",)
            else:
                next_state = (t, delim_leaked, token_guessed)

            action = (defense, t, "SUCCESS" if outcome else "FAIL", round(p_attack, 3))
            trace.append((state, action, next_state))
            state = next_state

            if outcome:
                success = True
                compromised += 1
                steps_to_compromise.append(t)
                hacked_by_step[t] += 1
                break

        traces.append(trace)

    return compromised / n, steps_to_compromise, hacked_by_step, traces


# ----------------------------
# Intrusion graph
# ----------------------------
def build_intrusion_graph(traces):
    """
    Intrusion graph:
    nodes: states
    edges: transitions labeled SUCCESS/FAIL (+ counts)
    This is safe with DiGraph now because SUCCESS goes to a distinct terminal node.
    """
    if not HAS_NX:
        return None

    G = nx.DiGraph()

    for trace in traces:
        for (s_from, action, s_to) in trace:
            label = action[2]  # SUCCESS / FAIL

            if not G.has_node(s_from):
                G.add_node(s_from)
            if not G.has_node(s_to):
                G.add_node(s_to)

            if G.has_edge(s_from, s_to) and G[s_from][s_to].get("label") == label:
                G[s_from][s_to]["count"] += 1
            else:
                G.add_edge(s_from, s_to, label=label, count=1)

    return G


def save_intrusion_graph_png(G, filename: str, title: str):
    if not HAS_NX or G is None:
        return

    plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(G, seed=SEED)

    nx.draw(G, pos, with_labels=True, node_size=900, font_size=7, arrows=True)
    edge_labels = {(u, v): f"{d['label']} ({d['count']})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(title)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ----------------------------
# Plotting
# ----------------------------
def plot_summary(results, K: int):
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)

    # Bars: static vs dynamic
    ax1 = fig.add_subplot(1, 2, 1)
    labels = DEFENSES
    static_vals = [results[d]["static_rate"] * 100.0 for d in DEFENSES]
    dynamic_vals = [results[d]["dynamic_rate"] * 100.0 for d in DEFENSES]

    x = list(range(len(labels)))
    w = 0.35
    ax1.bar([i - w/2 for i in x], static_vals, width=w, alpha=0.85, label="Static (1 try)")
    ax1.bar([i + w/2 for i in x], dynamic_vals, width=w, alpha=0.85, label=f"Dynamic (<= {K} tries)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Probability of Compromise (%)")
    ax1.set_title("Security Twin: Static Benchmark vs Dynamic Session Risk")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Cumulative curves
    ax2 = fig.add_subplot(1, 2, 2)
    xs = list(range(1, K + 1))
    for d in DEFENSES:
        ax2.plot(xs, results[d]["cumulative"], marker="o", markersize=4, linewidth=2, label=d)

    ax2.set_xlabel("Attempts in Session")
    ax2.set_ylabel("Cumulative P(Compromise) (%)")
    ax2.set_title(f"Compromise Probability vs Attempts (Max {K})")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_ylim(-2, 105)
    ax2.legend(fontsize=9)

    plt.savefig("spotlighting_security_twin_results.png")
    print("[OK] Saved: spotlighting_security_twin_results.png")


# ----------------------------
# Main
# ----------------------------
def main():
    random.seed(SEED)

    expected_collisions = NUM_SIMULATIONS * MAX_ATTEMPTS * COLLISION_PROB

    print("=== Spotlighting Security Twin (Summarization) ===")
    print("Scenario: Replicazione Paper vs Analisi Dinamica (Security Twin)")
    print(f"Simulazioni: {NUM_SIMULATIONS}, Max Tentativi/Episodio: {MAX_ATTEMPTS}")
    print(f"Prompt leak model (Delimiting): {ATTACKER_KNOWS_PROMPT}, P_leak/step={P_LEAK_PER_STEP:.3f}")
    print(f"Datamarking token-guess prob (1/N^k): {COLLISION_PROB:.2e}  | expected events ~ {expected_collisions:.2f}")
    print("-" * 60)

    results = {}

    for defense in DEFENSES:
        static_rate = run_static_test(defense, NUM_SIMULATIONS)
        dyn_rate, steps, hacked_by_step, traces = run_dynamic_twin(defense, NUM_SIMULATIONS, MAX_ATTEMPTS)

        avg_steps = statistics.mean(steps) if steps else float("nan")

        cumulative = []
        total = 0
        for t in range(1, MAX_ATTEMPTS + 1):
            total += hacked_by_step[t]
            cumulative.append(100.0 * total / NUM_SIMULATIONS)

        results[defense] = {
            "static_rate": static_rate,
            "dynamic_rate": dyn_rate,
            "avg_steps": avg_steps,
            "cumulative": cumulative,
            "traces": traces,
        }

        target = asr_static(defense) * 100.0
        print(f"=== DIFESA: {defense} ===")
        print(f"  Static ASR (Paper-like):     {static_rate*100:5.2f}%  (Target: {target:5.2f}%)")
        print(f"  Dynamic Session Risk (<=K):  {dyn_rate*100:5.2f}%")
        if steps:
            print(f"  Media tentativi per violare: {avg_steps:5.2f}")
        else:
            print("  Nessuna violazione rilevata.")
        print()

    plot_summary(results, MAX_ATTEMPTS)

    if HAS_NX:
        delim_traces = results["Delimiting"]["traces"]
        G = build_intrusion_graph(delim_traces)
        save_intrusion_graph_png(G, "intrusion_graph_delimiting.png",
                                 "Intrusion Graph (Toy) - Delimiting (Leak/Learning)")
        print("[OK] Saved: intrusion_graph_delimiting.png")
    else:
        print("[WARN] networkx non installato. Installa con: pip install networkx")


if __name__ == "__main__":
    main()

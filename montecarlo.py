import random
import math
import statistics
import matplotlib.pyplot as plt
import os
import glob

# Intrusion graph
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ----------------------------
# CONFIG
SEED = 42
NUM_SIMULATIONS = 5000
MAX_ATTEMPTS = 20

# --- Delimiting leak model (toy) ---
ATTACKER_KNOWS_PROMPT = True
LEARN_LAMBDA = 0.35
P_LEAK_PER_STEP = 1.0 - math.exp(-LEARN_LAMBDA)

# --- Datamarking guess model (toy) ---
N_CHARS = 62
K_GRAM = 3 #Paper used 5
COLLISION_PROB = 1.0 / (N_CHARS ** K_GRAM)

DEFENSES = ["None", "InstructionsOnly", "Delimiting", "Datamarking", "Encoding"]

# ----------------------------
# ASR inputs
MODELS_ASR = {
    # Paper ASR
    "GPT-3.5": {
        "None": 0.60,
        "InstructionsOnly": 0.58,
        "Delimiting": 0.30,
        "Datamarking": 0.031,
        "Encoding": 0.0,
    },
    # ASR estimated
    "Qwen 2.5": {
        "None": 0.45,
        "InstructionsOnly": 0.25,
        "Delimiting": 0.55,   # worse than baseline
        "Datamarking": 0.10,
        "Encoding": 0.0,
    },
    # ASR estimated
    "Mistral": {
        "None": 0.15,
        "InstructionsOnly": 0.0,
        "Delimiting": 0.0,
        "Datamarking": 0.0,
        "Encoding": 0.0,
    },
}


def asr_static(defense: str, model_name: str) -> float:
    return MODELS_ASR[model_name][defense]


# ----------------------------
# Cleanup old PNGs
def cleanup_old_pngs():
    patterns = [
        "security_twin_results_*.png",
        "intrusion_graph_delimiting_*.png",
        "spotlighting_security_twin_results.png",
        "intrusion_graph_delimiting.png",
    ]
    deleted = 0
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                os.remove(f)
                deleted += 1
            except OSError as e:
                print(f"[WARN] Could not delete {f}: {e}")
    print(f"[OK] Deleted {deleted} old .png files.")


# ----------------------------
# Simulations
def run_static_test(defense: str, n: int, model_name: str) -> float:
    """Static benchmark: 1 attempt per episode (single-shot ASR)."""
    p = asr_static(defense, model_name)
    wins = 0
    for _ in range(n):
        wins += (random.random() < p)
    return wins / n


def should_collapse_to_baseline(model_name: str, defense: str) -> bool:
    """
    Collapse logic:
    - GPT-3.5 and Mistral: allow collapse (worst-case attacker learns/guesses)
    - Qwen: allow collapse ONLY if the defense is actually better than baseline
           (otherwise reverting to baseline would paradoxically 'improve' security)
    """
    if model_name != "Qwen 2.5":
        return True

    p_baseline = asr_static("None", model_name)
    p_defended = asr_static(defense, model_name)
    return p_defended < p_baseline


def run_dynamic_twin(defense: str, n: int, K: int, model_name: str, save_traces: bool = False):
    """
    Dynamic Security Twin:
    - Up to K attempts per episode.
    - Compromised if ANY attempt succeeds (session risk).
    - Delimiting: stochastic leak event; after leak, optionally revert to baseline.
    - Datamarking: rare token-guess event; after guess, optionally revert to baseline.
    - Encoding: fixed at 0.
    """
    compromised = 0
    steps_to_compromise = []
    hacked_by_step = [0] * (K + 1)
    traces = [] if save_traces else None

    p_baseline = asr_static("None", model_name)
    p_defended = asr_static(defense, model_name)

    for _ in range(n):
        delim_leaked = False
        token_guessed = False

        state = (0, delim_leaked, token_guessed)
        trace = [] if save_traces else None

        for t in range(1, K + 1):
            # Update attacker knowledge
            if defense == "Delimiting" and ATTACKER_KNOWS_PROMPT and not delim_leaked and t >= 2:
                if random.random() < P_LEAK_PER_STEP:
                    delim_leaked = True

            if defense == "Datamarking" and not token_guessed:
                if random.random() < COLLISION_PROB:
                    token_guessed = True

            # Determine per-attempt success probability
            p_attack = p_defended

            # Collapse only if enabled for this model/defense
            if defense == "Delimiting" and delim_leaked and should_collapse_to_baseline(model_name, "Delimiting"):
                p_attack = p_baseline

            if defense == "Datamarking" and token_guessed and should_collapse_to_baseline(model_name, "Datamarking"):
                p_attack = p_baseline

            if defense == "Encoding":
                p_attack = 0.0

            outcome = (random.random() < p_attack)

            # traces (for intrusion graph)
            if save_traces:
                next_state = ("COMPROMISED",) if outcome else (t, delim_leaked, token_guessed)
                action = (defense, t, "SUCCESS" if outcome else "FAIL", float(p_attack))
                trace.append((state, action, next_state))
                state = next_state

            if outcome:
                compromised += 1
                steps_to_compromise.append(t)
                hacked_by_step[t] += 1
                break

        if save_traces:
            traces.append(trace)

    return compromised / n, steps_to_compromise, hacked_by_step, traces


# ----------------------------
# Intrusion graph (aggregated, correct counts)
def build_intrusion_graph(traces):
    """
    Robust aggregation: for each (u -> v) we accumulate:
      - count, success, fail, p_sum, p_avg
    (No overwriting problems as with DiGraph+single label.)
    """
    if not HAS_NX or not traces:
        return None

    G = nx.DiGraph()

    for trace in traces:
        for (s_from, action, s_to) in trace:
            outcome = action[2]          # "SUCCESS" / "FAIL"
            p = float(action[3])

            if not G.has_node(s_from):
                G.add_node(s_from)
            if not G.has_node(s_to):
                G.add_node(s_to)

            if not G.has_edge(s_from, s_to):
                G.add_edge(s_from, s_to, count=0, success=0, fail=0, p_sum=0.0)

            e = G[s_from][s_to]
            e["count"] += 1
            e["p_sum"] += p
            if outcome == "SUCCESS":
                e["success"] += 1
            else:
                e["fail"] += 1

    for u, v, d in G.edges(data=True):
        d["p_avg"] = d["p_sum"] / max(1, d["count"])
        d["label"] = f"c:{d['count']} S:{d['success']} F:{d['fail']} p_avg:{d['p_avg']:.2f}"

    return G


# ----------------------------
# Ordered layout
def ordered_grid_layout(G, x_scale: float = 1.8, y_scale: float = 1.6):
    """
    Deterministic, explanatory layout:
      x = time step t
      y = state class rows
    COMPROMISED at the far right.
    """
    def is_compromised(s):
        return isinstance(s, tuple) and len(s) == 1 and s[0] == "COMPROMISED"

    # rows (top -> bottom)
    y_map = {
        (False, False): 3,  # CLEAN
        (True,  False): 2,  # LEAK
        (False, True):  1,  # GUESS -> In Delimitng does not GUESS
        (True,  True):  0,  # BOTH
    }

    max_t = 0
    for n in G.nodes():
        if not is_compromised(n):
            t, leaked, guessed = n
            max_t = max(max_t, t)

    pos = {}
    for n in G.nodes():
        if is_compromised(n):
            pos[n] = ((max_t + 2) * x_scale, 1.5 * y_scale)
        else:
            t, leaked, guessed = n
            pos[n] = (t * x_scale, y_map[(leaked, guessed)] * y_scale)

    return pos


# ----------------------------
# Save intrusion graph PNG
def save_intrusion_graph_png(
    G,
    filename: str,
    title: str,
    label_top_k: int = 18,
    always_label_compromised_in: bool = True,
    show_vertical_step_guides: bool = True
):
    """
    Draws ALL relevant transitions (no edge filtering), but keeps it readable by:
      - ordered grid layout (no spaghetti)
      - labels only on the most frequent edges (+ always those into COMPROMISED)
      - slight curvature on edges to reduce overlaps
      - optional vertical guides for steps
    """
    if not HAS_NX or G is None:
        return

    def is_compromised(s):
        return isinstance(s, tuple) and len(s) == 1 and s[0] == "COMPROMISED"

    def pretty_state(s):
        if is_compromised(s):
            return "COMPROMISED"
        t, leaked, guessed = s
        if leaked and guessed:
            return f"t={t}\nBOTH"
        if leaked:
            return f"t={t}\nLEAK"
        if guessed:
            return f"t={t}\nGUESS"
        return f"t={t}\nCLEAN"

    # Ordered positions
    pos = ordered_grid_layout(G)

    # Determine max_t for guides and sizing
    max_t = 0
    for n in G.nodes():
        if not is_compromised(n):
            max_t = max(max_t, n[0])

    # Node styling
    node_labels = {n: pretty_state(n) for n in G.nodes()}
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if is_compromised(n):
            node_colors.append("red")
            node_sizes.append(1700)
        else:
            t, leaked, guessed = n
            if leaked and guessed:
                node_colors.append("gold")
            elif leaked:
                node_colors.append("orange")
            elif guessed:
                node_colors.append("yellow")
            else:
                node_colors.append("lightblue")
            node_sizes.append(900)

    # Edge list and styling
    edges_all = list(G.edges())
    total = sum(G[u][v].get("count", 0) for u, v in edges_all) or 1
    max_c = max((G[u][v].get("count", 0) for u, v in edges_all), default=1)

    edge_widths = []
    edge_colors = []
    for u, v in edges_all:
        d = G[u][v]
        c = d.get("count", 0)
        # width: gentle scaling
        w = 0.8 + 4.5 * (c / max_c)
        edge_widths.append(w)
        edge_colors.append("green" if d.get("success", 0) > 0 else "gray")

    # Choose which edges to label
    edges_sorted = sorted(
        ((u, v, G[u][v].get("count", 0)) for (u, v) in edges_all),
        key=lambda x: x[2],
        reverse=True
    )

    edges_to_label = set((u, v) for (u, v, _) in edges_sorted[:label_top_k])

    if always_label_compromised_in:
        for (u, v) in edges_all:
            if is_compromised(v):
                edges_to_label.add((u, v))

    edge_labels = {(u, v): G[u][v].get("label", "") for (u, v) in edges_to_label}

    # Draw
    plt.figure(figsize=(20, 9))

    # Optional vertical guides per step
    if show_vertical_step_guides:
        xs = sorted(set(x for (x, y) in pos.values()))
        # The x positions are scaled steps (t * x_scale); build lines for t=0..max_t
        # We infer x_scale from t=1 - t=0 when possible
        inferred_xs = {}
        for n, (x, y) in pos.items():
            if not is_compromised(n):
                inferred_xs[n[0]] = x
        for t in range(0, max_t + 1):
            if t in inferred_xs:
                x = inferred_xs[t]
                plt.axvline(x=x, linewidth=0.6, alpha=0.25)
                plt.text(x, -0.6, f"t={t}", fontsize=8, ha="center", alpha=0.8)

        # Row labels on the left
        x_min = min(x for x, y in pos.values())
        plt.text(x_min - 1.5, 3 * 1.6, "CLEAN", fontsize=10, alpha=0.9)
        plt.text(x_min - 1.5, 2 * 1.6, "LEAK", fontsize=10, alpha=0.9)
        plt.text(x_min - 1.5, 1 * 1.6, "GUESS", fontsize=10, alpha=0.9)
        plt.text(x_min - 1.5, 0 * 1.6, "BOTH", fontsize=10, alpha=0.9)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_all,
        arrows=True,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.65,
        connectionstyle="arc3,rad=0.08"
    )

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        bbox=dict(alpha=0.7, edgecolor="none")
    )

    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ----------------------------
# Plotting
def plot_summary(results, K: int, model_name: str):
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)

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
    ax1.set_title(f"Security Twin: {model_name} (Static vs Dynamic)")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = fig.add_subplot(1, 2, 2)
    xs = list(range(1, K + 1))
    for d in DEFENSES:
        ax2.plot(xs, results[d]["cumulative"], marker="o", markersize=4, linewidth=2, label=d)

    ax2.set_xlabel("Attempts in Session")
    ax2.set_ylabel("Cumulative P(Compromise) (%)")
    ax2.set_title(f"{model_name}: Risk Evolution (Max {K} steps)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_ylim(-2, 105)
    ax2.legend(fontsize=9)

    safe_name = model_name.replace(" ", "_").lower()
    out = f"security_twin_results_{safe_name}.png"
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved: {out}")


# ----------------------------
# Main
def main():
    random.seed(SEED)
    cleanup_old_pngs()

    expected_collisions = NUM_SIMULATIONS * MAX_ATTEMPTS * COLLISION_PROB

    print("=== Spotlighting Security Twin (Summarization) ===")
    print(f"Simulations: {NUM_SIMULATIONS}, Max attempts/episode: {MAX_ATTEMPTS}")
    print(f"Delimiting leak model: knows_prompt={ATTACKER_KNOWS_PROMPT}, P_leak/step={P_LEAK_PER_STEP:.3f}")
    print(f"Datamarking guess prob: {COLLISION_PROB:.2e} | expected events ~ {expected_collisions:.2f}")
    print("-" * 70)

    for model_name in MODELS_ASR.keys():
        print(f"\n>>> MODEL: {model_name} <<<")

        results = {}

        base = asr_static("None", model_name)
        for d in DEFENSES:
            if d != "None" and asr_static(d, model_name) > base:
                print(f"[WARN] {model_name}: {d} ASR ({asr_static(d, model_name):.3f}) > baseline ({base:.3f})")

        for defense in DEFENSES:
            static_rate = run_static_test(defense, NUM_SIMULATIONS, model_name)

            # Save traces ONLY for Delimiting
            save_traces = (defense == "Delimiting")
            dyn_rate, steps, hacked_by_step, traces = run_dynamic_twin(
                defense, NUM_SIMULATIONS, MAX_ATTEMPTS, model_name, save_traces=save_traces
            )

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

            target = asr_static(defense, model_name) * 100.0
            print(f"--- Defense: {defense}")
            print(f"  Static ASR:             {static_rate*100:5.2f}% (target input: {target:5.2f}%)")
            print(f"  Dynamic session risk:   {dyn_rate*100:5.2f}%")
            if steps:
                print(f"  Avg steps to compromise: {avg_steps:5.2f}")
            else:
                print("  No compromises observed.")
            print()

        plot_summary(results, MAX_ATTEMPTS, model_name)

        if HAS_NX:
            delim_traces = results["Delimiting"]["traces"]
            G = build_intrusion_graph(delim_traces)
            safe_name = model_name.replace(" ", "_").lower()
            save_intrusion_graph_png(
                G,
                f"intrusion_graph_delimiting_{safe_name}.png",
                f"Intrusion Graph (Toy) - Delimiting - {model_name}",
                label_top_k=18,                
                always_label_compromised_in=True,
                show_vertical_step_guides=True 
            )
            print(f"[OK] Saved: intrusion_graph_delimiting_{safe_name}.png")
        else:
            print("[WARN] networkx not installed (skip intrusion graph).")

        print("-" * 70)


if __name__ == "__main__":
    main()

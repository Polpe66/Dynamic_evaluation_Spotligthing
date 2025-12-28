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
P_LEAK_PER_STEP = 1.0 - math.exp(-LEARN_LAMBDA)

# --- CONFIGURAZIONE MODELLI ---
# Qui inseriamo le probabilità statiche (Attack Success Rate - ASR) per ogni modello
MODELS_ASR = {
    "GPT-3.5": {
        "None": 0.60,
        "InstructionsOnly": 0.60,
        "Delimiting": 0.30,
        "Datamarking": 0.031,
        "Encoding": 0.0,
    },
    "Qwen 2.5": {
        "None": 0.45,            # Baseline
        "InstructionsOnly": 0.25, # Migliora un po' rispetto a baseline
        "Delimiting": 0.55,       # ATTENZIONE: Peggiora (Attention Trap)
        "Datamarking": 0.10,      # Molto efficace, ma non zero
        "Encoding": 0.0,
    }
}

DEFENSES = ["None", "InstructionsOnly", "Delimiting", "Datamarking", "Encoding"]


# ----------------------------
# Helpers
# ----------------------------
def asr_static(defense: str, model_name: str) -> float:
    """Restituisce l'ASR statico per la difesa e il modello specificati."""
    return MODELS_ASR[model_name][defense]


# ----------------------------
# Simulation
# ----------------------------
def run_static_test(defense: str, n: int, model_name: str) -> float:
    """Static benchmark: 1 attempt per episode (paper-like single-shot ASR)."""
    p = asr_static(defense, model_name)
    wins = 0
    for _ in range(n):
        wins += (random.random() < p)
    return wins / n


def run_dynamic_twin(defense: str, n: int, K: int, model_name: str):
    """
    Dynamic Security Twin simulation.
    """
    compromised = 0
    steps_to_compromise = []
    hacked_by_step = [0] * (K + 1)
    traces = []

    for _ in range(n):
        # Twin state variables
        delim_leaked = False
        token_guessed = False

        state = (0, delim_leaked, token_guessed)  # (attempt_index, delim_leaked, token_guessed)
        trace = []

        success = False

        for t in range(1, K + 1):
            # --- Update attacker knowledge / rare events (before attempt) ---
            if defense == "Delimiting" and ATTACKER_KNOWS_PROMPT and not delim_leaked and t >= 2:
                if random.random() < P_LEAK_PER_STEP:
                    delim_leaked = True

            if defense == "Datamarking" and not token_guessed:
                if random.random() < COLLISION_PROB:
                    token_guessed = True

            # --- Determine effective per-attempt success probability ---
            p_attack = asr_static(defense, model_name)

            # If a defense collapses, revert to baseline of THAT model
            if defense == "Delimiting" and delim_leaked:
                p_attack = asr_static("None", model_name)

            if defense == "Datamarking" and token_guessed:
                p_attack = asr_static("None", model_name)

            if defense == "Encoding":
                p_attack = 0.0

            # --- Monte Carlo attempt ---
            outcome = (random.random() < p_attack)

            # Intrusion-graph next state
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
def plot_summary(results, K: int, model_name: str):
    """
    Genera un grafico specifico per il modello passato come argomento.
    """
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    
    # Clean model name for filename
    safe_name = model_name.replace(" ", "_").lower()

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
    ax1.set_title(f"Security Twin: {model_name} (Static vs Dynamic)")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Cumulative curves
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

    filename = f"security_twin_results_{safe_name}.png"
    plt.savefig(filename)
    print(f"[OK] Saved: {filename}")
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    random.seed(SEED)

    expected_collisions = NUM_SIMULATIONS * MAX_ATTEMPTS * COLLISION_PROB

    print("=== Spotlighting Security Twin (Multi-Model) ===")
    print(f"Simulazioni: {NUM_SIMULATIONS}, Max Tentativi/Episodio: {MAX_ATTEMPTS}")
    print(f"Modelli testati: {list(MODELS_ASR.keys())}")
    print("-" * 60)

    # Ciclo su ogni modello configurato
    for model_name in MODELS_ASR.keys():
        print(f"\n>>> AVVIO SIMULAZIONE PER MODELLO: {model_name.upper()} <<<")
        
        results = {}

        for defense in DEFENSES:
            # Passiamo model_name alle funzioni
            static_rate = run_static_test(defense, NUM_SIMULATIONS, model_name)
            dyn_rate, steps, hacked_by_step, traces = run_dynamic_twin(defense, NUM_SIMULATIONS, MAX_ATTEMPTS, model_name)

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
            print(f"--- DIFESA: {defense} ---")
            print(f"  Static ASR:               {static_rate*100:5.2f}%  (Input: {target:5.2f}%)")
            print(f"  Dynamic Risk (Session):   {dyn_rate*100:5.2f}%")
            if steps:
                print(f"  Media tentativi per violare: {avg_steps:5.2f}")
            else:
                print("  Nessuna violazione rilevata.")

        # Plot specifico per questo modello
        plot_summary(results, MAX_ATTEMPTS, model_name)

        # Generazione Intrusion Graph (solo per Delimiting che è il caso più interessante dinamicamente)
        if HAS_NX:
            delim_traces = results["Delimiting"]["traces"]
            G = build_intrusion_graph(delim_traces)
            safe_name = model_name.replace(" ", "_").lower()
            graph_filename = f"intrusion_graph_delimiting_{safe_name}.png"
            
            save_intrusion_graph_png(G, graph_filename,
                                     f"Intrusion Graph - Delimiting - {model_name}")
            print(f"[OK] Saved Graph: {graph_filename}")
        else:
            print("[WARN] networkx non installato.")
        
        print("-" * 40)

if __name__ == "__main__":
    main()
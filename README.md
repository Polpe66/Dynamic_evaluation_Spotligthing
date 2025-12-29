# Spotlighting Security Twin (Monte Carlo)

Toy implementation di un *Security Twin* per valutare dinamicamente il rischio di compromissione (session risk) in scenari di indirect prompt injection, confrontando difese Spotlighting.

Il codice usa valori di ASR statici (paper + stime manuali) e simula una sessione con fino a K tentativi tramite Monte Carlo, producendo:
- confronto **Static (1 try)** vs **Dynamic (≤K tries)**
- curva di rischio cumulativo
- intrusion graph (solo per Delimiting)

## Setup & Run
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/montecarlo.py




Il programma genera automaticamente i file:

security_twin_results_gpt-3.5.png

security_twin_results_mistral.png

security_twin_results_qwen_2.5.png


intrusion_graph_delimiting_gpt-3.5.png

intrusion_graph_delimiting_mistral.png

intrusion_graph_delimiting_qwen_2.5.png

Gli ASR di Qwen 2.5 e Mistral sono stati stimati manualmente con Ollama (20 tentativi per difesa, prompt di attacco uguali tra condizioni).

Il modello assume tentativi indipendenti (a parte l’evento di leak per Delimiting).

from flask import Flask, jsonify, render_template
from flask_cors import CORS
import random

# Import your Simulated Annealing Python script
# (If your file is named helpdesk_sa_2.py, change this to: import helpdesk_sa_2 as helpdesk_sa)
import helpdesk_sa

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    # Serves templates/index.html
    return render_template('index.html')

@app.route('/api/get_sequence', methods=['GET'])
def get_sequence():
    # 1. Generate a random valid starting order using your Python function
    rng = random.Random(3)
    start = helpdesk_sa.random_valid_order(rng)
    
    # 2. Run Simulated Annealing in Python
    best_seq, best_cost, hist = helpdesk_sa.simulated_annealing(start)
    sa_etr, sa_ef, sa_v = helpdesk_sa.breakdown(best_seq)
    
    # 3. Calculate Naive Order metrics from Python
    naive_seq = helpdesk_sa.NAIVE_ORDER
    naive_cost = helpdesk_sa.calculate_cost(naive_seq)
    n_etr, n_ef, n_v = helpdesk_sa.breakdown(naive_seq)

    return jsonify({
        "status": "success",
        "sa_optimized": {
            "sequence": best_seq,
            "cost": round(best_cost, 1),
            "etr": round(sa_etr, 1),
            "ef": round(sa_ef, 2),
            "violations": sa_v,
            "iterations": len(hist["current"]),
            "uphill_moves": hist["uphill"]
        },
        "naive": {
            "sequence": naive_seq,
            "cost": round(naive_cost, 1),
            "etr": round(n_etr, 1),
            "ef": round(n_ef, 2),
            "violations": n_v
        },
        "steps": helpdesk_sa.STEPS,
        "prerequisites": helpdesk_sa.PREREQUISITES,
        "w_frustration": helpdesk_sa.W_FRUSTRATION
    })

if __name__ == '__main__':
    print("Starting Helpdesk Triage Backend on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
# NexaBot: AI-Optimized IT Helpdesk Triage

**Author:** Aonjaneya Banerjee  
**Task:** Task 1 — Healthcare or Technology Project Using AI Tools  

---

## 🤖 Built with AI Tools
**This entire full-stack project—from the mathematical optimization logic to the responsive UI—was conceptualized, architected, and generated using advanced AI Large Language Models.** 

Leveraging AI tools allowed for rapid prototyping of a complex operations-research problem (Simulated Annealing) and seamless integration into a modern web stack within a highly compressed timeframe. AI was utilized for:
*   **Algorithmic Design:** Formulating the cost function and Simulated Annealing logic in Python.
*   **Full-Stack Development:** Generating the Flask API bridge and the vanilla JS frontend.
*   **UI/UX Design:** Crafting the Tailwind CSS layout, chat interface, and data-visualization modals.
*   **Documentation:** Generating the comprehensive PDF project report and this README.

---

## 📌 Project Overview
Enterprise IT helpdesks waste thousands of hours forcing users through static troubleshooting runbooks. NexaBot solves this combinatorial scheduling problem using a **Simulated Annealing** algorithm. 

Instead of guessing the best order to ask troubleshooting questions, NexaBot mathematically calculates the optimal sequence of diagnostic steps to minimize **Expected Time to Resolution (ETR)** and **User Frustration**, while strictly obeying technical prerequisite rules (e.g., checking a VPN before flushing a DNS cache).

The project features a **Python Flask backend** running the optimization algorithm, connected to a **Tailwind CSS + JS frontend** that simulates a live triage chat environment.

## 🧮 Mathematical Model (The AI Engine)
The algorithm evaluates the expected cost of any given sequence of questions $\pi = (\pi_1, ..., \pi_N)$. 

Because a user only reaches step $k$ if all previous steps failed, the reach probability is:
$R_k = \prod_{j=1}^{k-1} (1 - p_{\pi_j})$

The total Expected Cost $E(\pi)$ (in seconds-equivalent) minimizes time, frustration ($f$), and prerequisite violations ($V$):
$E(\pi) = \sum_{k=1}^{N} [R_k \cdot t_{\pi_k}] + w_f \sum_{k=1}^{N} [R_k \cdot f_{\pi_k}] + \Lambda \cdot V(\pi)$

## 🛠️ Tech Stack
*   **Backend:** Python 3, Flask, Flask-CORS, NumPy
*   **Algorithm:** Simulated Annealing (Metaheuristic Optimization)
*   **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript
*   **AI Assistance:** Google Gemini / LLMs

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/NexaBot-IT-Triage.git
   cd NexaBot-IT-Triage
   ```

2. **Install the required Python libraries:**
   ```bash
   pip install flask flask-cors numpy matplotlib
   ```

3. **Run the Flask Backend:**
   ```bash
   python app.py
   ```

4. **Access the Application:**
   Open your web browser and navigate to:
   `http://127.0.0.1:5000/`

## 📊 Results
Compared to a standard human-guessed IT runbook, the AI-generated Simulated Annealing sequence reduces expected diagnostic costs by **>14%**, deferring heavy, frustrating tasks (like reinstalling network drivers) to the absolute end of the queue.
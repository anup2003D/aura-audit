# Aura-Audit Intern Evaluation (180 Minutes)

Welcome to the summer internship evaluation. You are tasked with building a responsible, end-to-end customer support AI system.

## Project Flow
This is a **sequential** 11-step challenge. Each step builds on the previous one.

### Phase 1: Foundation (60 min)
1. **Normalize Data:** Clean 1,000 raw logs in `data/raw/`. Remove PII.
2. **Unsupervised Discovery:** Use K-Means to identify issue clusters. 
3. **Semi-supervised Labeling:** Propagate labels from clusters to the full set.
4. **Supervised Baseline:** Train a Random Forest intent classifier.

### Phase 2: Neural & RL (60 min)
5. **Neural Network:** Implement an MLP classifier.
6. **Reinforcement Learning:** Optimize a reward-based decision script (Q-Learning).
7. **In-processing Audit:** Apply re-weighting to mitigate bias found in training.

### Phase 3: Agents & Governance (60 min)
8. **RAG Pipeline:** Set up a FAISS Vector Store for log retrieval.
9. **Agentic Loop:** Implement a ReAct Agent that uses the classifier and RAG as tools.
10. **Post-processing Audit:** Run a final post-processing guardrail and SHAP explainability.
11. **Compliance:** Complete the Model Card and Impact Assessment.

## Requirements
- Use **Random Seed 42** globally.
- Ensure your code is modular (Foundation, Intelligence, Governance).
- You are graded on **Logic**, **Understanding**, and **Auditing Depth**.

## Submission
Fork the project and update the code and share the link to lakshminarasimhan.santhanam@qli.org.in

You should complete ASAP.  Kindly note that Entries will be considered First in First Out.

Good luck!

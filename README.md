# Self-Healing Text Classification DAG

This repository implements a self-healing sentiment classification pipeline using a fine-tuned DistilBERT model integrated into a LangGraph-style Directed Acyclic Graph (DAG). The DAG intelligently routes predictions through confidence checks and a fallback mechanism to ensure correctness, particularly in low-confidence scenarios.

---

##  Key Features

* **Fine-Tuned Transformer Model:** Trained on Google GoEmotions (3-way: Positive, Negative, Neutral)
* **Self-Healing DAG:** Built with LangGraph-style nodes (Inference, ConfidenceCheck, Fallback)
* **Confidence Thresholding:** Prevents blind automation by triggering fallback when confidence < 80%
* **CLI Interaction:** Clean loop with human-in-the-loop clarification
* **Structured Logging:** Records predictions, confidence, fallback triggers, and final decisions
* **Visual Insights:** CLI and Matplotlib plots for fallback and confidence analysis
* **Optional Backup Model:** Zero-shot fallback logic using `facebook/bart-large-mnli` (Bonus)

---

##  Training Summary

* **Model:** DistilBERT (fine-tuned)
* **Dataset:** Subset of [GoEmotions](https://github.com/google-research/goemotions)
* **Epochs:** 3
* **Accuracy (Val):** 75.2%
* **F1 Macro (Val):** 74.99%
* **Training Loss:**  0.57
* Confidence Threshold: 0.8 (80%) — Configured explicitly in ConfidenceCheckNode(threshold=0.8) — This threshold ensures the model only auto-accepts high-confidence predictions. If confidence drops below 80%, ConfidenceCheckNode(threshold=0.8) triggers FallbackNode, allowing human intervention.
---

##  Folder Structure

```
.
Self-Healing-Text-Classification-DAG.zip        # All files are zipped for submission
└── Self-Healing-Text-Classification-DAG/
    ├── main.py                     # Main CLI interface
    ├── log_viewer.py               # Utility to view structured logs
    ├── requirements.txt            # Python dependencies
    ├── model_output/               # Fine-tuned DistilBERT model
    ├── distilbert-base-uncased/    # Original base model files
    ├── dag_modules/
    │   ├── inference_node.py       # Inference node
    │   ├── confidence_node.py      # Confidence check logic
    │   └── fallback_node.py        # Human-in-the-loop fallback handler
    ├── selfhealing/
    │   ├── dag_log_*.jsonl         # Structured log files
    │   ├── confidence_plot.png     # Visual of confidence scores
    │   ├── fallback_histogram.png  # Fallback frequency chart
    │   └── fallback_cli_histogram.py # CLI fallback frequency summary
    ├── optional/
    │   ├── fine_tune.py              # Training script (optional, not executed)
    │   ├── main_backupmodel.py      # Backup model demo logic (not executed)
    │   └── fallback_nodebackupmodel.py
    ├── GoEmotions.ipynb           # Notebook containing model outputs and metrics
    ├── Self_Healing_Text_Classification_DAG.pdf  # Final presentation
    └── DEMO_Self_Healing_Text_Classification_DAG.mp4
    ├── optional/                     # Experimental backup model fallback
    │   ├── main_backupmodel.py
    │   └── fallback_nodebackupmodel.py
    │   └── fine_tune.py              # Training script (optional) 
             
   NOTE: fine_tune.py : This script fine-tunes a DistilBERT model on 3 sentiment labels (positive, negative, neutral) for potential integration as the primary or fallback classifier within the DAG. Execution was not completed due to memory constraints, but the pipeline is fully  defined and ready to run with transformers, datasets, and evaluate. Same holds good for main_backupmodel.py and fallback_nodebackupmodel.py , please consider these files as conceptual/integrative but not executed.
```

---

##  Setup Instructions

```bash
# Create and activate a new environment (optional but recommended)
python -m venv selfheal-env
source selfheal-env/bin/activate  # On Windows: selfheal-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

##  How to Run the CLI

```bash
python main.py
```

### Example:

```
Input: Sort of liked it... but not really sure.
[InferenceNode] Predicted label: Positive | Confidence: 50%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a positive review?
User: no
Please select the correct label:
0: Positive
1: Negative
2: Neutral
Your selection: 2
Final Label: Neutral (Corrected via user clarification)
```

---

##  Logs

* Structured logs saved in `logs/*.jsonl`
* Captures:

  * Timestamps
  * Input text
  * Initial prediction (label + confidence)
  * Fallback trigger status
  * Final decision source
  * User clarification (if any)

Use `python log_viewer.py` to print a summary.

---

##  Self-Healing

* **Fallback Histogram:** Distribution of fallback labels
* **Confidence Plot:** Variability in prediction confidence
* **CLI Summary:** Fallback triggers + manual corrections

Located in `selfhealing/`

---

##  Bonus: Backup Model (Zero-Shot)

* **Model:** `facebook/bart-large-mnli`
* **Triggered When:** Confidence < 40% (optional setup)
* **Run With:** `main_backupmodel.py`
* **FallbackNode:** `fallback_nodebackupmodel.py`

```python
from transformers import pipeline
backup = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
backup("I don't know", candidate_labels=["Positive", "Neutral", "Negative"])
```


---

##  Demo Deliverables

* [x] `main.py` CLI with DAG + fallback
* [x] Fine-tuned model: `model_output/`
* [x] Structured logs + visual plots
* [x] Optional zero-shot backup fallback
* [x] README + 2–3 minute PDF deck + demo video


---

##  Author

* Author: Alekya Rani Seerapu
* Date: June 25, 2025

---

##  License

* For assignment use only.

---

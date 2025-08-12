import json
import time
import os
from datetime import datetime

# Configuration files
LOG_FILE = "cpu.logs"
PREDICTION_FILE = "predictions.txt"
CONFIG_FILE = "config.json"

# Global state
previous_issues = set()
model_stats = {
    "Model 1": {"accuracy": 0.4928},
    "Model 2": {"accuracy": 0.4932},
    "Model 3": {"accuracy": 0.4926},
    "Model 4": {"accuracy": 0.4974},
}
current_model = "Model 2"  # Default
auto_switching = True

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def load_config():
    global current_model, auto_switching
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            auto_switching = config.get("auto_switching", True)
            current_model = config.get("manual_model", "Model 2")
            log_message(f"Config Loaded | Auto: {auto_switching} | Model: {current_model}")
    except Exception as e:
        log_message(f"⚠️ Error loading config: {str(e)} — Using defaults")

def load_data():
    try:
        with open("data.json", "r") as f:
            input_json = json.load(f)
    except Exception as e:
        log_message(f"Error loading data: {str(e)}")
        return []

    if isinstance(input_json, dict) and "evaluation" in input_json and "next_issue_prediction" in input_json:
        model_mapping = {
            "LogisticRegression": "Model 1",
            "HoeffdingTree": "Model 2",
            "NaiveBayes": "Model 3",
            "BaggingClassifier": "Model 4"
        }

        eval_models = input_json["evaluation"]["models"]
        eval_entry = {
            "issue": input_json["evaluation"]["issue_id"],
            "actual": eval_models[0]["actual_class"] if eval_models else None,
            "predictions": {}
        }

        for model in eval_models:
            name = model_mapping.get(model["name"])
            if name:
                eval_entry["predictions"][name] = model["predicted_class"]

        next_models = input_json["next_issue_prediction"]["models"]
        next_entry = {
            "issue": input_json["next_issue_prediction"]["issue_id"],
            "predictions": {},
            "confidence": {}
        }

        for model in next_models:
            name = model_mapping.get(model["name"])
            if name:
                next_entry["predictions"][name] = model["predicted_class"]
                try:
                    conf_value = float(model["confidence"].strip('%')) / 100.0
                except:
                    conf_value = 0.0
                next_entry["confidence"][name] = conf_value

        return [eval_entry, next_entry]

    return input_json

def evaluate_models(latest_issue):
    global current_model
    report = []
    losses = {}

    for model, stats in model_stats.items():
        actual = latest_issue.get("actual")
        pred = latest_issue.get("predictions", {}).get(model)

        if actual is None or pred is None:
            continue

        result = "✅ WIN" if pred == actual else "❌ LOSS"
        losses[model] = 0 if pred == actual else 1

        report.append(f"🔹 {model.upper()} ➜ PRED: {pred.upper()} | ACTUAL: {actual.upper()} | RESULT: {result} | ACCURACY: {stats['accuracy']*100:.2f}%")

    if auto_switching:
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        if current_model in losses and losses[current_model] == 1:
            for name, _ in sorted_models:
                if name != current_model and losses.get(name, 0) == 0:
                    log_message(f"Auto-Switching model from {current_model} to {name}")
                    current_model = name
                    break
            else:
                current_model = sorted_models[0][0]
    else:
        log_message(f"Manual Mode Active - Using Fixed Model: {current_model}")

    return report, current_model

def get_latest_complete():
    data = load_data()
    for entry in reversed(data):
        if "actual" in entry:
            return entry
    return None

def get_next_prediction_entry():
    data = load_data()
    for entry in reversed(data):
        if "actual" not in entry:
            return entry
    return None

def write_prediction(report):
    with open(PREDICTION_FILE, "w") as f:
        f.write(report)
    log_message(f"Updated prediction file: {PREDICTION_FILE}")

def main():
    log_message("🎯 AI SYSTEM STARTED")
    load_config()

    while True:
        try:
            latest_result = get_latest_complete()
            next_prediction = get_next_prediction_entry()

            if latest_result and next_prediction:
                issue_prev = latest_result.get("issue")
                issue_next = next_prediction.get("issue")
                pred = next_prediction["predictions"].get(current_model, "UNKNOWN")
                conf = next_prediction.get("confidence", {}).get(current_model, 0.0)
                acc = model_stats.get(current_model, {}).get("accuracy", 0) * 100

                evaluation_report, selected_model = evaluate_models(latest_result)

                prediction_output = f"""🎯 PREDICTION REPORT
🆔 ISSUE: {issue_prev}

📊 LAST RESULT EVALUATION:
{chr(10).join(evaluation_report)}

📈 NEXT PREDICTION 🐞
🆔 ISSUE: {issue_next}
🔥 USING: {selected_model.upper()}
🔮 PREDICTION: {pred.upper()}
📊 CONFIDENCE: {conf:.2f} | ACCURACY: {acc:.2f}%

🔄 Updated in real-time...
"""
                write_prediction(prediction_output)
                log_message(f"Processed issue {issue_prev} -> {issue_next}")

        except Exception as e:
            log_message(f"⚠️ Error: {str(e)}")

        time.sleep(2)

if __name__ == "__main__":
    with open(LOG_FILE, "w") as f:
        f.write("")
    main()
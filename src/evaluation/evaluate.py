import time
import numpy as np
import yaml
import torch

class EvaluationReport:
    """
    Computes confusion matrix, classification metrics, and throughput performance.
    """
    def __init__(self):
        self.latencies = []
        self.predictions = []
        self.ground_truths = []

    def log_result(self, pred_class, gt_class, latency_ms):
        self.predictions.append(pred_class)
        self.ground_truths.append(gt_class)
        self.latencies.append(latency_ms)

    def generate_report(self):
        preds = np.array(self.predictions)
        gts = np.array(self.ground_truths)
        
        total = len(gts)
        correct = np.sum(preds == gts)
        accuracy = (correct / total) * 100.0 if total > 0 else 0.0
        
        # Calculate positive-class metrics (defect vs no-defect)
        # Class 0: Background/Pass, 1..N: Defect/Fail
        preds_binary = (preds > 0).astype(int)
        gts_binary = (gts > 0).astype(int)
        
        tp = np.sum((preds_binary == 1) & (gts_binary == 1))
        fp = np.sum((preds_binary == 1) & (gts_binary == 0))
        fn = np.sum((preds_binary == 0) & (gts_binary == 1))
        tn = np.sum((preds_binary == 0) & (gts_binary == 0))
        
        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_latency = np.mean(self.latencies) if len(self.latencies) > 0 else 0.0
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        
        print("\n" + "="*50)
        print("         SECURECOATING-VISION EVALUATION REPORT")
        print("="*50)
        print(f"Total Samples Evaluated : {total}")
        print(f"Overall Accuracy        : {accuracy:.2f}%")
        print(f"Defect Precision        : {precision:.2f}%")
        print(f"Defect Recall (Sens.)   : {recall:.2f}% (Target: >=98.2%)")
        print(f"Defect F1-Score         : {f1_score:.2f}%")
        print(f"Average Latency         : {avg_latency:.2f} ms (Target: <=35ms)")
        print(f"System Throughput       : {throughput:.2f} FPS")
        print("="*50 + "\n")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_latency_ms": avg_latency,
            "throughput_fps": throughput
        }

def run_evaluation(config_path="configs/model.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    num_classes = config.get("model", {}).get("num_classes", 5)
    
    print("Initializing offline validation run...")
    eval_logger = EvaluationReport()
    
    # Simulate a validation loop on 20 dummy instances
    np.random.seed(42)
    for i in range(20):
        # Defect occurrence bias for testing (40% defect rate)
        gt = int(np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.1, 0.1, 0.1, 0.1]))
        
        # Simulate predictions with high overlap (adding minor random classification noise)
        pred = gt if np.random.rand() < 0.90 else int(np.random.choice(range(num_classes)))
        
        # Simulate processing time (20-30ms range)
        latency_ms = np.random.uniform(22.0, 31.0)
        
        eval_logger.log_result(pred, gt, latency_ms)
        
    metrics = eval_logger.generate_report()
    return metrics

if __name__ == "__main__":
    run_evaluation()

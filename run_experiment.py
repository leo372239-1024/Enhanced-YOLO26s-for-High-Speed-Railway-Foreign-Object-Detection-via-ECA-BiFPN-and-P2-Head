import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import ultralytics.nn.tasks as ultralytics_tasks
from ultralytics.nn.modules.conv import Concat
import shutil
from modules.eca_attention import ECAAttention

# NumPy >=2.4 removed np.trapz; Ultralytics 8.3.x still calls it during validation.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

# Register custom modules so Ultralytics YAML parser can resolve these names.
ultralytics_tasks.ECAAttention = ECAAttention
# Fallback to parser-compatible concat blocks for BiFPN layer names in YAML.
ultralytics_tasks.BiFPN_Concat = Concat
ultralytics_tasks.BiFPN_Concat_3 = Concat

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "dataset/RailFOD23/data.yaml")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

def check_env():
    print("="*50)
    print("Environment Check")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics not found. Please install it: pip install ultralytics")
        sys.exit(1)
    print("="*50)

def train_model(model_cfg, model_name, weights=None, epochs=100, batch=8):
    print(f"\n>>> Training {model_name}...")
    model = YOLO(model_cfg)
    if weights and os.path.exists(weights):
        model.load(weights)
    
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        name=model_name,
        project=RUNS_DIR,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        workers=2,
        exist_ok=True,
        amp=True
    )
    return results

def test_model(model_path, model_name):
    print(f"\n>>> Testing {model_name}...")
    model = YOLO(model_path)
    results = model.val(
        data=DATA_YAML,
        split='test',
        project=RUNS_DIR,
        name=f"{model_name}_test",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        exist_ok=True
    )
    return results

def generate_comparison_plots(metrics_df):
    print("\n>>> Generating Comparison Plots...")
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. mAP Comparison Bar Plot
    plt.figure(figsize=(10, 6))
    metrics_melted = metrics_df.melt(id_vars='Model', value_vars=['mAP50', 'mAP50-95'])
    sns.barplot(x='Model', y='value', hue='variable', data=metrics_melted)
    plt.title('mAP Comparison')
    plt.ylabel('Score')
    plt.savefig(os.path.join(plots_dir, 'map_comparison.png'))
    plt.close()

    # 2. Precision/Recall Comparison
    plt.figure(figsize=(10, 6))
    metrics_melted = metrics_df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1'])
    sns.barplot(x='Model', y='value', hue='variable', data=metrics_melted)
    plt.title('Precision, Recall, F1 Comparison')
    plt.ylabel('Score')
    plt.savefig(os.path.join(plots_dir, 'pr_f1_comparison.png'))
    plt.close()

def collect_results(baseline_res, improved_res):
    print("\n>>> Collecting Results...")
    data = []
    
    # Baseline metrics
    b_m = baseline_res.results_dict
    data.append({
        'Model': 'YOLO26s_Baseline',
        'Precision': b_m.get('metrics/precision(B)', 0),
        'Recall': b_m.get('metrics/recall(B)', 0),
        'F1': 2 * (b_m.get('metrics/precision(B)', 0) * b_m.get('metrics/recall(B)', 0)) / (b_m.get('metrics/precision(B)', 0) + b_m.get('metrics/recall(B)', 0) + 1e-6),
        'mAP50': b_m.get('metrics/mAP50(B)', 0),
        'mAP50-95': b_m.get('metrics/mAP50-95(B)', 0)
    })
    
    # Improved metrics
    i_m = improved_res.results_dict
    data.append({
        'Model': 'YOLO26s_ECA_BiFPN_P2',
        'Precision': i_m.get('metrics/precision(B)', 0),
        'Recall': i_m.get('metrics/recall(B)', 0),
        'F1': 2 * (i_m.get('metrics/precision(B)', 0) * i_m.get('metrics/recall(B)', 0)) / (i_m.get('metrics/precision(B)', 0) + i_m.get('metrics/recall(B)', 0) + 1e-6),
        'mAP50': i_m.get('metrics/mAP50(B)', 0),
        'mAP50-95': i_m.get('metrics/mAP50-95(B)', 0)
    })
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    df.to_excel(os.path.join(RESULTS_DIR, "metrics.xlsx"), index=False)
    print(df)
    return df

def main():
    check_env()
    
    EPOCHS = 100 
    
    baseline_cfg = os.path.join(BASE_DIR, "models/yolo26s_baseline.yaml")
    baseline_weights = os.path.join(WEIGHTS_DIR, "yolo26s.pt")
    
    # 1. Baseline Training
    train_model(baseline_cfg, "baseline", weights=baseline_weights, epochs=EPOCHS)
    
    # 2. Baseline Testing
    baseline_best = os.path.join(RUNS_DIR, "baseline/weights/best.pt")
    baseline_test_res = test_model(baseline_best, "baseline")
    
    # 3. Improved Model Training
    improved_cfg = os.path.join(BASE_DIR, "models/yolo26s_ECA_BiFPN_P2.yaml")
    train_model(improved_cfg, "improved", weights=baseline_weights, epochs=EPOCHS)
    
    # 4. Improved Model Testing
    improved_best = os.path.join(RUNS_DIR, "improved/weights/best.pt")
    improved_test_res = test_model(improved_best, "improved")
    
    # 5. Indicators Comparison
    metrics_df = collect_results(baseline_test_res, improved_test_res)
    
    # 6. Auto Generate Paper Images
    generate_comparison_plots(metrics_df)
    
    # Copy important plots from runs
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(RUNS_DIR, "improved/results.png")):
        shutil.copy(os.path.join(RUNS_DIR, "improved/results.png"), os.path.join(plots_dir, "train_loss_curve.png"))
    if os.path.exists(os.path.join(RUNS_DIR, "improved/PR_curve.png")):
        shutil.copy(os.path.join(RUNS_DIR, "improved/PR_curve.png"), os.path.join(plots_dir, "precision_recall_curve.png"))
    if os.path.exists(os.path.join(RUNS_DIR, "improved/F1_curve.png")):
        shutil.copy(os.path.join(RUNS_DIR, "improved/F1_curve.png"), os.path.join(plots_dir, "F1_curve.png"))
    
    # Visualization samples
    pred_dir = os.path.join(RESULTS_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    test_run_dir = os.path.join(RUNS_DIR, "improved_test")
    if os.path.exists(test_run_dir):
        val_batch_labels = [f for f in os.listdir(test_run_dir) if f.startswith('val_batch0_labels')]
        val_batch_preds = [f for f in os.listdir(test_run_dir) if f.startswith('val_batch0_pred')]
        
        if val_batch_labels:
            shutil.copy(os.path.join(test_run_dir, val_batch_labels[0]), os.path.join(pred_dir, "gt_samples.jpg"))
        if val_batch_preds:
            shutil.copy(os.path.join(test_run_dir, val_batch_preds[0]), os.path.join(pred_dir, "pred_samples.jpg"))

    print("\n" + "="*50)
    print("Experiment Completed Successfully!")
    print(f"Results are saved in: {RESULTS_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()

from fairlearn.metrics import selection_rate, demographic_parity_difference, MetricFrame
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def audit_for_bias(y_true, y_pred, sensitive_features):
    """
    Task 3.1: Calculate fairness metrics using Fairlearn.
    - Measure Demographic Parity Difference.
    - Measure Selection Rate across groups.
    """
    print(f"\n{'='*70}")
    print("BIAS AUDIT: FAIRNESS METRICS")
    print(f"{'='*70}")
    
    # Calculate Demographic Parity Difference
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    print(f"Demographic Parity Difference: {dpd:.4f}")
    print(f"  (Closer to 0 is better, range: [-1, 1])")
    
    # Calculate Selection Rate across groups
    print(f"\nSelection Rates by Group:")
    unique_groups = np.unique(sensitive_features)
    for group in unique_groups:
        mask = sensitive_features == group
        group_pred = y_pred[mask]
        sel_rate = selection_rate(y_true[mask], group_pred)
        print(f"  {group}: {sel_rate:.4f}")
    
    # Use MetricFrame for detailed analysis
    from sklearn.metrics import accuracy_score
    metric_frame = MetricFrame(
        metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    print(f"\nDetailed Metrics by Group:")
    print(metric_frame.by_group)
    
    print(f"\nFairness Assessment:")
    if abs(dpd) < 0.1:
        print(f"  ✓ Model shows good demographic parity (DPD < 0.1)")
    elif abs(dpd) < 0.2:
        print(f"  ⚠ Model shows moderate bias (0.1 <= DPD < 0.2)")
    else:
        print(f"  ✗ Model shows significant bias (DPD >= 0.2)")
    
    return dpd, metric_frame

def explain_decisions(model, X_sample):
    """
    Task 3.1: Use SHAP to explain model predictions.
    - Generate summary_plot.
    """
    import matplotlib.pyplot as plt
    import os
    
    print(f"\n{'='*70}")
    print("MODEL EXPLAINABILITY: SHAP ANALYSIS")
    print(f"{'='*70}")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For multi-class, take the first class for visualization
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[0]
    else:
        shap_values_plot = shap_values
    
    print(f"Generated SHAP values for {len(X_sample)} samples")
    
    # Generate summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_plot, X_sample, show=False)
    
    # Save plot
    output_dir = "../../aura_audit/data/processed"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ SHAP summary plot saved to: {plot_path}")
    plt.close()
    
    return explainer, shap_values

def apply_post_processing_mitigation(y_pred, sensitive_features):
    """
    Task 3.1: Implement a Post-processing guardrail.
    - Adjust thresholds to equalize odds or selection rates.
    """
    print(f"\n{'='*70}")
    print("POST-PROCESSING MITIGATION")
    print(f"{'='*70}")
    
    # Calculate current selection rates
    unique_groups = np.unique(sensitive_features)
    group_rates = {}
    
    for group in unique_groups:
        mask = sensitive_features == group
        group_rates[group] = np.mean(y_pred[mask])
    
    print(f"Original selection rates:")
    for group, rate in group_rates.items():
        print(f"  {group}: {rate:.4f}")
    
    # Calculate target rate (average across all groups)
    target_rate = np.mean(list(group_rates.values()))
    print(f"\nTarget rate (average): {target_rate:.4f}")
    
    # Adjust predictions to match target rate per group
    y_pred_adjusted = y_pred.copy()
    
    for group in unique_groups:
        mask = sensitive_features == group
        group_preds = y_pred[mask]
        current_rate = group_rates[group]
        
        if current_rate > target_rate:
            # Reduce positive predictions
            n_to_flip = int((current_rate - target_rate) * len(group_preds))
            positive_indices = np.where((mask) & (y_pred == 1))[0]
            if len(positive_indices) > n_to_flip:
                flip_indices = np.random.choice(positive_indices, n_to_flip, replace=False)
                y_pred_adjusted[flip_indices] = 0
        elif current_rate < target_rate:
            # Increase positive predictions
            n_to_flip = int((target_rate - current_rate) * len(group_preds))
            negative_indices = np.where((mask) & (y_pred == 0))[0]
            if len(negative_indices) > n_to_flip:
                flip_indices = np.random.choice(negative_indices, n_to_flip, replace=False)
                y_pred_adjusted[flip_indices] = 1
    
    # Calculate new selection rates
    print(f"\nAdjusted selection rates:")
    for group in unique_groups:
        mask = sensitive_features == group
        new_rate = np.mean(y_pred_adjusted[mask])
        print(f"  {group}: {new_rate:.4f}")
    
    print(f"✓ Post-processing mitigation applied")
    
    return y_pred_adjusted

def train_fair_model(X_train, y_train, sensitive_features_train):
    """
    In-processing Fairness: Train a fair model using re-weighting.
    - Use Fairlearn's ExponentiatedGradient with DemographicParity constraint
    - Apply sample re-weighting to ensure regional parity
    """
    print(f"\n{'='*70}")
    print("IN-PROCESSING FAIRNESS: TRAINING FAIR MODEL")
    print(f"{'='*70}")
    
    # Calculate sample weights to balance sensitive groups
    unique_groups = np.unique(sensitive_features_train)
    group_counts = {group: np.sum(sensitive_features_train == group) for group in unique_groups}
    max_count = max(group_counts.values())
    
    # Create weights (inverse frequency weighting)
    sample_weights = np.zeros(len(sensitive_features_train))
    for group in unique_groups:
        mask = sensitive_features_train == group
        weight = max_count / group_counts[group]
        sample_weights[mask] = weight
    
    print(f"Sample weights by region:")
    for group in unique_groups:
        mask = sensitive_features_train == group
        avg_weight = np.mean(sample_weights[mask])
        print(f"  {group}: {avg_weight:.4f} (count: {group_counts[group]})")
    
    # Train baseline model without fairness constraints
    print(f"\nTraining baseline model (no fairness constraints)...")
    baseline_model = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=50)
    baseline_model.fit(X_train, y_train)
    
    # Train fair model with re-weighting
    print(f"Training fair model with sample re-weighting...")
    fair_model = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=50)
    fair_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Train model with Fairlearn's ExponentiatedGradient
    print(f"Training model with ExponentiatedGradient + DemographicParity...")
    try:
        base_estimator = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=50, max_depth=5)
        constraint = DemographicParity()
        
        mitigator = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint,
            eps=0.01,
            max_iter=50
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)
        print(f"✓ ExponentiatedGradient model trained successfully")
    except Exception as e:
        print(f"⚠ ExponentiatedGradient failed (compatibility issue): {e}")
        print(f"  Using re-weighted model as the mitigated model")
        mitigator = None
    
    print(f"✓ Fair models trained successfully")
    
    return baseline_model, fair_model, mitigator

if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer
    import os
    
    print("="*70)
    print("FAIRNESS AUDIT & IN-PROCESSING MITIGATION")
    print("="*70)
    
    # Load processed data
    data_path = "../../aura_audit/data/processed/support_logs_cleaned.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run pipeline.py first.")
        exit(1)
    
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} logs")
    
    # Prepare features and labels
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['cluster'].values
    sensitive_features = df['region'].values
    
    # Split data
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Task 3.1: Train models with and without fairness constraints
    baseline_model, fair_model, mitigator = train_fair_model(X_train, y_train, sf_train)
    
    # Evaluate all models
    print(f"\n{'='*70}")
    print("MODEL EVALUATION")
    print(f"{'='*70}")
    
    # Baseline predictions
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    print(f"\nBaseline Model Accuracy: {baseline_acc:.4f}")
    
    # Fair model (re-weighted) predictions
    y_pred_fair = fair_model.predict(X_test)
    fair_acc = accuracy_score(y_test, y_pred_fair)
    print(f"Fair Model (Re-weighted) Accuracy: {fair_acc:.4f}")
    
    # Mitigated model predictions
    if mitigator is not None:
        y_pred_mitigated = mitigator.predict(X_test)
        mitigated_acc = accuracy_score(y_test, y_pred_mitigated)
        print(f"Fair Model (ExponentiatedGradient) Accuracy: {mitigated_acc:.4f}")
    else:
        y_pred_mitigated = y_pred_fair
        mitigated_acc = fair_acc
        print(f"Fair Model (ExponentiatedGradient) Accuracy: {mitigated_acc:.4f} (using re-weighted)")
    
    # Audit baseline model for bias
    print(f"\n{'='*70}")
    print("BASELINE MODEL BIAS AUDIT")
    print(f"{'='*70}")
    dpd_baseline, metrics_baseline = audit_for_bias(y_test, y_pred_baseline, sf_test)
    
    # Audit fair model (re-weighted) for bias
    print(f"\n{'='*70}")
    print("FAIR MODEL (RE-WEIGHTED) BIAS AUDIT")
    print(f"{'='*70}")
    dpd_fair, metrics_fair = audit_for_bias(y_test, y_pred_fair, sf_test)
    
    # Audit mitigated model for bias
    print(f"\n{'='*70}")
    print("FAIR MODEL (EXPONENTIATED GRADIENT) BIAS AUDIT")
    print(f"{'='*70}")
    dpd_mitigated, metrics_mitigated = audit_for_bias(y_test, y_pred_mitigated, sf_test)
    
    # Compare fairness improvements
    print(f"\n{'='*70}")
    print("FAIRNESS COMPARISON")
    print(f"{'='*70}")
    print(f"Model                          | Accuracy | DPD (abs)")
    print(f"-" * 70)
    print(f"Baseline (no mitigation)       | {baseline_acc:.4f}   | {abs(dpd_baseline):.4f}")
    print(f"Fair (re-weighted)             | {fair_acc:.4f}   | {abs(dpd_fair):.4f}")
    print(f"Fair (ExponentiatedGradient)   | {mitigated_acc:.4f}   | {abs(dpd_mitigated):.4f}")
    
    print(f"\nFairness Improvement:")
    print(f"  Re-weighted DPD reduction: {abs(dpd_baseline) - abs(dpd_fair):.4f}")
    print(f"  ExponentiatedGradient DPD reduction: {abs(dpd_baseline) - abs(dpd_mitigated):.4f}")
    
    # SHAP explainability on the best fair model
    best_model = mitigator if (mitigator is not None and abs(dpd_mitigated) < abs(dpd_fair)) else fair_model
    model_name = "ExponentiatedGradient" if best_model == mitigator else "Re-weighted"
    
    print(f"\n{'='*70}")
    print(f"EXPLAINABILITY: BEST FAIR MODEL ({model_name})")
    print(f"{'='*70}")
    
    # Use a sample for SHAP (computational efficiency)
    X_sample = X_test[:100]
    if model_name == "Re-weighted":
        explain_decisions(best_model, X_sample)
    else:
        print("Note: SHAP analysis skipped for ExponentiatedGradient (use base estimator)")
    
    # Apply post-processing mitigation to baseline
    print(f"\n{'='*70}")
    print("POST-PROCESSING MITIGATION ON BASELINE")
    print(f"{'='*70}")
    y_pred_post_processed = apply_post_processing_mitigation(y_pred_baseline, sf_test)
    dpd_post, _ = audit_for_bias(y_test, y_pred_post_processed, sf_test)
    
    # Create results summary
    results = pd.DataFrame({
        'model': ['Baseline', 'Fair (Re-weighted)', 'Fair (ExponentiatedGradient)', 'Post-processed'],
        'accuracy': [baseline_acc, fair_acc, mitigated_acc, accuracy_score(y_test, y_pred_post_processed)],
        'demographic_parity_diff': [abs(dpd_baseline), abs(dpd_fair), abs(dpd_mitigated), abs(dpd_post)]
    })
    
    print(f"\n{'='*70}")
    print("✓ In-processing fairness mitigation complete!")
    print(f"{'='*70}")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def run_supervised_baseline(X, y):
    """
    Task 1.2: Train a Supervised Random Forest.
    - Split data (test_size=0.2, random_state=RANDOM_SEED)
    - Train RandomForestClassifier(random_state=RANDOM_SEED)
    - Return accuracy and report
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Train Random Forest
    clf = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print("SUPERVISED BASELINE (RANDOM FOREST)")
    print(f"{'='*70}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)
    
    return accuracy, report, clf

def run_semi_supervised_learning(X_unlabelled, X_labelled, y_labelled):
    """
    Task 1.3: Use LabelSpreading to propagate labels to the unlabelled set.
    - Combine labelled and unlabelled data
    - Set labels for unlabelled as -1
    - Fit LabelSpreading
    - Return the full pseudo-labelled dataset
    """
    import numpy as np
    
    # Combine labelled and unlabelled data
    X_combined = np.vstack([X_labelled, X_unlabelled])
    
    # Create labels array: labelled data gets actual labels, unlabelled gets -1
    y_combined = np.concatenate([y_labelled, np.full(len(X_unlabelled), -1)])
    
    print(f"\n{'='*70}")
    print("SEMI-SUPERVISED LEARNING (LABEL SPREADING)")
    print(f"{'='*70}")
    print(f"Labelled samples: {len(X_labelled)}")
    print(f"Unlabelled samples: {len(X_unlabelled)}")
    print(f"Total samples: {len(X_combined)}")
    
    # Fit LabelSpreading
    label_spread = LabelSpreading(kernel='knn', n_neighbors=7, max_iter=30, alpha=0.2)
    label_spread.fit(X_combined, y_combined)
    
    # Get predicted labels for all data
    y_predicted = label_spread.transduction_
    
    # Extract predictions for originally unlabelled data
    y_unlabelled_predicted = y_predicted[len(X_labelled):]
    
    # Calculate label distribution
    unique_labels, counts = np.unique(y_unlabelled_predicted, return_counts=True)
    
    print(f"\nLabel propagation complete!")
    print(f"Predicted label distribution for unlabelled data:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} samples ({count/len(y_unlabelled_predicted)*100:.1f}%)")
    
    # Return the full pseudo-labelled dataset
    return y_predicted, label_spread

def run_neural_network(X, y):
    """
    Task 1.4: Train a Neural Network (MLP) classifier.
    - Split data (test_size=0.2, random_state=RANDOM_SEED)
    - Train MLPClassifier with hidden layers
    - Return accuracy and report for comparison with baseline
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Train Neural Network (MLP)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = mlp.predict(X_test)
    accuracy = mlp.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print("NEURAL NETWORK (MLP)")
    print(f"{'='*70}")
    print(f"Architecture: {mlp.hidden_layer_sizes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Iterations: {mlp.n_iter_}")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)
    
    return accuracy, report, mlp

if __name__ == "__main__":
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    import os
    
    # Load cleaned data with clusters
    print("Loading processed data with clusters...")
    data_path = "../../aura_audit/data/processed/support_logs_cleaned.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} logs with {df['cluster'].nunique()} clusters\n")
    
    # Map clusters to intent labels based on dominant patterns
    # Analyze cluster characteristics to assign meaningful labels
    print(f"{'='*70}")
    print("LABEL GENERATION: Mapping Clusters to Intent Categories")
    print(f"{'='*70}")
    
    # Define intent mapping (cluster -> intent label)
    # Based on the data generation patterns:
    cluster_to_intent = {
        0: 'tech_support',      # Application crashes
        1: 'login_issue',       # Password/login problems
        2: 'billing',           # Invoice/payment issues
        3: 'refund_request',    # Refund requests
        4: 'general_inquiry'    # Other/mixed
    }
    
    # Assign intent labels based on clusters
    df['intent'] = df['cluster'].map(cluster_to_intent)
    
    print("\nCluster to Intent Mapping:")
    for cluster, intent in cluster_to_intent.items():
        count = len(df[df['cluster'] == cluster])
        print(f"  Cluster {cluster} -> {intent}: {count} samples")
    
    # Prepare features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['cluster'].values
    
    # Task 1.2: Run supervised baseline with all labelled data
    print(f"\n{'='*70}")
    print("TASK 1.2: SUPERVISED BASELINE")
    print(f"{'='*70}")
    rf_accuracy, rf_report, clf = run_supervised_baseline(X, y)
    
    # Task 1.4: Run Neural Network and compare with baseline
    print(f"\n{'='*70}")
    print("TASK 1.4: NEURAL NETWORK CLASSIFICATION")
    print(f"{'='*70}")
    mlp_accuracy, mlp_report, mlp = run_neural_network(X, y)
    
    # Comparison
    print(f"\n{'='*70}")
    print("MODEL COMPARISON: RANDOM FOREST vs NEURAL NETWORK")
    print(f"{'='*70}")
    print(f"Random Forest Accuracy:  {rf_accuracy:.4f}")
    print(f"Neural Network Accuracy: {mlp_accuracy:.4f}")
    print(f"Difference:              {mlp_accuracy - rf_accuracy:+.4f}")
    if mlp_accuracy > rf_accuracy:
        print(f"✓ Neural Network outperforms Random Forest by {(mlp_accuracy - rf_accuracy)*100:.2f}%")
    elif mlp_accuracy < rf_accuracy:
        print(f"✓ Random Forest outperforms Neural Network by {(rf_accuracy - mlp_accuracy)*100:.2f}%")
    else:
        print("✓ Both models perform equally")
    
    # Task 1.3: Semi-supervised learning simulation
    # Create a scenario where only 20% of data is labelled
    print(f"\n{'='*70}")
    print("TASK 1.3: SEMI-SUPERVISED LEARNING")
    print(f"{'='*70}")
    
    # Randomly select 20% as labelled, 80% as unlabelled
    n_labelled = int(0.2 * len(X))
    indices = np.arange(len(X))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    labelled_idx = indices[:n_labelled]
    unlabelled_idx = indices[n_labelled:]
    
    X_labelled = X[labelled_idx]
    y_labelled = y[labelled_idx]
    X_unlabelled = X[unlabelled_idx]
    y_unlabelled_true = y[unlabelled_idx]  # True labels (for evaluation only)
    
    # Run semi-supervised learning
    y_all_predicted, label_spread = run_semi_supervised_learning(X_unlabelled, X_labelled, y_labelled)
    
    # Evaluate semi-supervised predictions on unlabelled data
    y_unlabelled_predicted = y_all_predicted[len(X_labelled):]
    from sklearn.metrics import accuracy_score
    ss_accuracy = accuracy_score(y_unlabelled_true, y_unlabelled_predicted)
    
    print(f"\n{'='*70}")
    print("SEMI-SUPERVISED EVALUATION")
    print(f"{'='*70}")
    print(f"Accuracy on unlabelled data: {ss_accuracy:.4f}")
    print("\nClassification Report on Unlabelled Data:")
    print(classification_report(y_unlabelled_true, y_unlabelled_predicted))
    
    # Add predictions to dataframe
    df['predicted_intent'] = label_spread.transduction_[:len(df)]
    
    print(f"\n{'='*70}")
    print(f"✓ Semi-supervised learning complete!")
    print(f"{'='*70}")

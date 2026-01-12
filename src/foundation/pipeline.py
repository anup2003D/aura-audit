import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42

def clean_text(text):
    """
    Task 1.1: Implement text normalization and PII removal.
    - Convert to lowercase
    - Remove email addresses and phone numbers
    - Strip extra whitespace
    """
    import re
    
    if pd.isna(text):
        return text
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove urgent markers and excessive punctuation
    text = re.sub(r'!{3,}.*?!{3,}', '', text)  # Remove !!! patterns
    text = re.sub(r'urgent help needed', '', text, flags=re.IGNORECASE)
    
    # Remove PII - Email addresses
    text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '[EMAIL]', text)
    
    # Remove PII - Phone numbers (various formats)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    
    # Remove PII - PIN codes
    text = re.sub(r'pin code:\s*\d+\s*\(pii example\)', '[PIN]', text, flags=re.IGNORECASE)
    text = re.sub(r'pin:\s*\d+', '[PIN]', text, flags=re.IGNORECASE)
    
    # Normalize excessive punctuation
    text = re.sub(r'[!]{2,}', '.', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    # Strip extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def discover_intents(df, n_clusters=5):
    """
    Task 1.1: Use K-Means to identify clusters of support issues.
    - Convert text to numeric (hint: use TF-IDF or simple embeddings)
    - Apply KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    - Return the cluster labels
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Convert text to numeric using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df['clean_text'])
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Store the vectorizer and cluster centers for later analysis
    df['cluster'] = labels
    
    # Print cluster statistics
    print(f"\n{'='*70}")
    print("INTENT DISCOVERY (K-MEANS CLUSTERING)")
    print(f"{'='*70}")
    print(f"Number of clusters: {n_clusters}")
    print(f"\nCluster distribution:")
    print(df['cluster'].value_counts().sort_index().to_string())
    
    # Show sample texts from each cluster
    print(f"\n{'='*70}")
    print("SAMPLE TEXTS FROM EACH CLUSTER")
    print(f"{'='*70}")
    for i in range(n_clusters):
        cluster_samples = df[df['cluster'] == i]['clean_text'].head(2)
        print(f"\nCluster {i} ({len(df[df['cluster'] == i])} logs):")
        for j, text in enumerate(cluster_samples):
            print(f"  [{j+1}] {text[:80]}...")
    
    return labels

def plot_clusters(df, labels):
    """
    Task 1.1: Use PCA to visualize clusters in 2D.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import os
    
    # Convert text to numeric for PCA
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Customer Intent Clusters (K-Means + PCA Visualization)')
    plt.grid(True, alpha=0.3)
    
    print(f"\nPCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    import os
    
    # Load raw data
    raw_path = "aura-audit/aura_audit/data/raw/support_logs.csv"
    print(f"Loading raw logs from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} logs\n")
    
    # Task 1.1: Preprocessing data - Apply text normalization and PII removal
    print("Preprocessing data (Task 1.1: NLP normalization and PII removal)...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Generate normalization statistics
    print("\n" + "="*70)
    print("NORMALIZATION SUMMARY")
    print("="*70)
    print(f"Total logs processed: {len(df)}")
    
    # Count PII removals
    email_count = df['clean_text'].str.contains('[EMAIL]', regex=False).sum()
    phone_count = df['clean_text'].str.contains('[PHONE]', regex=False).sum()
    pin_count = df['clean_text'].str.contains('[PIN]', regex=False).sum()
    
    print(f"\nPII instances detected and removed:")
    print(f"  - Email addresses: {email_count}")
    print(f"  - Phone numbers: {phone_count}")
    print(f"  - PIN codes: {pin_count}")
    print(f"  - Total PII removed: {email_count + phone_count + pin_count}")
    
    # Show transformation examples
    print("\n" + "="*70)
    print("EXAMPLE TRANSFORMATIONS")
    print("="*70)
    pii_examples = df[df['clean_text'].str.contains(r'\[EMAIL\]|\[PHONE\]|\[PIN\]', regex=True)]
    if len(pii_examples) > 0:
        for i, (idx, row) in enumerate(pii_examples.head(3).iterrows()):
            print(f"\n[Example {i+1}] {row['log_id']}")
            print(f"BEFORE: {row['text'][:90]}...")
            print(f"AFTER:  {row['clean_text'][:90]}...")
    
    # Save cleaned data
    processed_dir = "../../aura_audit/data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "support_logs_cleaned.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned data saved to: {output_path}")
    
    print("\n" + "="*70)
    print("✓ Normalization complete! Ready for Task 1.1 intent discovery.")
    print("="*70)
    
    # Task 1.1: Unsupervised Intent Discovery using K-Means
    print("\nStarting unsupervised intent discovery...")
    labels = discover_intents(df, n_clusters=5)
    plot_clusters(df, labels)
    
    # Update the saved dataset with clusters
    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print("✓ Intent discovery complete! Dataset updated with cluster labels.")
    print(f"{'='*70}")
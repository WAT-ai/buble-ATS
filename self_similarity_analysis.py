import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from pathlib import Path
import seaborn as sns

def compute_cosine_similarity_matrix(chromagram):
    """
    Compute cosine similarity matrix from chromagram.
    
    Args:
        chromagram (np.ndarray): CENS chromagram (12 x time_frames)
    
    Returns:
        np.ndarray: Similarity matrix (time_frames x time_frames)
    """
    # Transpose to get time_frames x 12
    chroma_t = chromagram.T
    
    # Normalize each time frame vector
    chroma_normalized = chroma_t / np.linalg.norm(chroma_t, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(chroma_normalized, chroma_normalized.T)
    
    return similarity_matrix

def find_natural_gap_threshold(similarity_matrix):
    """
    Find natural gap in similarity distribution for threshold.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix
    
    Returns:
        float: Threshold value at the largest gap
    """
    # Get upper triangle (avoid diagonal and duplicates)
    # issue might be here because the column is top to bottom descending rather then bottom to top ascending
    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    
    # Sort similarities
    sorted_similarities = np.sort(upper_triangle)
    
    # Find largest gap
    differences = np.diff(sorted_similarities)
    max_gap_index = np.argmax(differences)
    
    # Threshold is value just before the gap
    threshold = sorted_similarities[max_gap_index]
    
    print(f"Natural gap threshold: {threshold:.3f}")
    return threshold

def hierarchical_clustering_with_silhouette(similarity_matrix, max_clusters=10):
    """
    Perform hierarchical clustering with adaptive silhouette score optimization.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix
        max_clusters (int): Maximum number of clusters to try
    
    Returns:
        tuple: (best_clusters, best_cutoff, best_score)
    """
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Try different cutoff distances - focus on balanced clusters
    cutoffs = np.linspace(0.2, 0.9, 25)  # Wider range, more granular
    best_score = -1
    best_cutoff = 0.4
    best_clusters = None
    
    print("Optimizing clustering with silhouette score...")
    
    for cutoff in cutoffs:
        # Get clusters for this cutoff
        clusters = fcluster(linkage_matrix, cutoff, criterion='distance')
        n_clusters = len(np.unique(clusters))
        
        # Prefer balanced clusters (3-6) for better section detection
        if n_clusters < 3 or n_clusters > 6:
            continue
        
        # Check cluster balance - avoid one very large cluster
        cluster_sizes = [len(np.where(clusters == i)[0]) for i in np.unique(clusters)]
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        balance_ratio = min_size / max_size if max_size > 0 else 0
        
        # Skip if one cluster is too dominant (less than 20% balance)
        if balance_ratio < 0.2:
            continue
        
        # Calculate silhouette score
        try:
            score = silhouette_score(distance_matrix, clusters, metric='precomputed')
            
            # Prefer solutions with better balance and fewer clusters
            if score > best_score or (score == best_score and n_clusters < len(np.unique(best_clusters))):
                best_score = score
                best_cutoff = cutoff
                best_clusters = clusters
                
        except ValueError:
            # Skip if silhouette score can't be calculated
            continue
    
    # If no good clustering found, use a fallback approach
    if best_clusters is None:
        print("Using fallback clustering approach...")
        # Try to force more balanced clustering
        for cutoff in [0.3, 0.4, 0.5, 0.6]:
            clusters = fcluster(linkage_matrix, cutoff, criterion='distance')
            n_clusters = len(np.unique(clusters))
            if 3 <= n_clusters <= 6:
                cluster_sizes = [len(np.where(clusters == i)[0]) for i in np.unique(clusters)]
                balance_ratio = min(cluster_sizes) / max(cluster_sizes)
                if balance_ratio >= 0.15:  # Slightly more lenient
                    best_clusters = clusters
                    best_cutoff = cutoff
                    break
        
        if best_clusters is None:
            # Last resort: use a very aggressive cutoff
            best_cutoff = 0.7
            best_clusters = fcluster(linkage_matrix, best_cutoff, criterion='distance')
        
        best_score = 0.0  # Placeholder
    
    print(f"Best silhouette score: {best_score:.3f}")
    print(f"Best cutoff distance: {best_cutoff:.3f}")
    print(f"Number of clusters: {len(np.unique(best_clusters))}")
    
    # Print cluster balance information
    cluster_sizes = [len(np.where(best_clusters == i)[0]) for i in np.unique(best_clusters)]
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Balance ratio: {min(cluster_sizes) / max(cluster_sizes):.3f}")
    
    return best_clusters, best_cutoff, best_score

def extract_section_features(similarity_matrix, clusters, sr, hop_length):
    """
    Extract comprehensive features for each detected section.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix
        clusters (np.ndarray): Cluster assignments
        sr (int): Sample rate
        hop_length (int): Hop length
    
    Returns:
        np.ndarray: Feature array with all section information
    """
    unique_clusters = np.unique(clusters)
    sections = []
    
    print(f"Processing {len(unique_clusters)} unique clusters...")
    
    for cluster_id in unique_clusters:
        # Find all time frames in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        
        # Skip clusters that are too small (less than 50 frames = ~2.3 seconds)
        if len(cluster_indices) < 50:
            print(f"Skipping cluster {cluster_id} - too small ({len(cluster_indices)} frames)")
            continue
        
        # Convert frame indices to time
        start_time = cluster_indices[0] * hop_length / sr
        end_time = cluster_indices[-1] * hop_length / sr
        length = end_time - start_time
        
        # Skip sections that are too short (less than 10 seconds)
        if length < 10.0:
            print(f"Skipping cluster {cluster_id} - too short ({length:.1f}s)")
            continue
        
        # Calculate relative position (0.0-1.0)
        total_duration = similarity_matrix.shape[0] * hop_length / sr
        position = start_time / total_duration
        
        # Calculate similarity to cluster centroid
        cluster_similarities = similarity_matrix[cluster_indices][:, cluster_indices]
        avg_similarity = np.mean(cluster_similarities)
        
        # Calculate frequency (how many times this section appears)
        frequency = len(cluster_indices)
        
        # Calculate similarity score (how similar to rest of cluster)
        # Use average similarity to other points in same cluster
        cluster_similarity_scores = []
        for i in cluster_indices:
            same_cluster_similarities = similarity_matrix[i, cluster_indices]
            cluster_similarity_scores.append(np.mean(same_cluster_similarities))
        
        similarity_score = np.mean(cluster_similarity_scores)
        
        # Add section features
        sections.append([
            start_time,      # Start time (seconds)
            end_time,        # End time (seconds)
            length,          # Duration (seconds)
            position,        # Relative position (0.0-1.0)
            similarity_score, # Similarity to cluster
            cluster_id,      # Section ID (number)
            frequency,       # Number of occurrences
            avg_similarity   # Average similarity within cluster
        ])
        
        print(f"  Cluster {cluster_id}: {start_time:.1f}s - {end_time:.1f}s (length: {length:.1f}s)")
    
    return np.array(sections)

def plot_similarity_matrix_with_sections(similarity_matrix, clusters, sr, hop_length, save_path=None):
    """
    Create comprehensive visualization of similarity matrix with detected sections.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix
        clusters (np.ndarray): Cluster assignments
        sr (int): Sample rate
        hop_length (int): Hop length
        save_path (str): Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Similarity matrix heatmap
    im = ax1.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    ax1.invert_yaxis()
    
    # Add time annotations
    n_frames = similarity_matrix.shape[0]
    time_points = np.linspace(0, n_frames * hop_length / sr, 6)
    frame_points = time_points * sr / hop_length
    
    ax1.set_xticks(frame_points)
    ax1.set_xticklabels([f'{t:.0f}s' for t in time_points])
    ax1.set_yticks(frame_points)
    ax1.set_yticklabels([f'{t:.0f}s' for t in time_points])
    
    ax1.set_title('Self-Similarity Matrix with Detected Sections')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Time (B to T)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax1, label='Similarity')
    
    # 2. Color-coded sections
    unique_clusters = np.unique(clusters)
    
    # Filter out clusters that are too small
    valid_clusters = []
    for cluster_id in unique_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) >= 30:  # Reduced minimum size threshold
            valid_clusters.append(cluster_id)
    
    if len(valid_clusters) == 0:
        # If no valid clusters, show original clustering but limit to top clusters
        cluster_sizes = [(cluster_id, len(np.where(clusters == cluster_id)[0])) 
                        for cluster_id in unique_clusters]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        valid_clusters = [cluster_id for cluster_id, size in cluster_sizes[:5]]  # Top 5 largest
        print(f"Warning: No clusters met size requirements, showing top {len(valid_clusters)} largest clusters")
    
    # Create a mapping from cluster IDs to sequential indices for better visualization
    cluster_to_index = {cluster_id: i for i, cluster_id in enumerate(valid_clusters)}
    
    # Create temporal section map (shows section type at each time point)
    temporal_sections = np.zeros(similarity_matrix.shape[0], dtype=int)
    for cluster_id in valid_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        temporal_sections[cluster_indices] = cluster_to_index[cluster_id] + 1
    
    # Create a 1D temporal visualization showing section sequence
    # This will be displayed as a horizontal bar
    temporal_bar = temporal_sections.reshape(1, -1)
    
    # Use a standard colormap with proper normalization
    im2 = ax2.imshow(temporal_bar, cmap='tab10', aspect='auto', vmin=0, vmax=len(valid_clusters))
    
    # Add section boundary lines for temporal visualization
    for i, cluster_id in enumerate(valid_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) > 1:
            # Add vertical lines at section boundaries
            start_idx = cluster_indices[0]
            end_idx = cluster_indices[-1]
            
            # Vertical lines at section boundaries
            ax2.axvline(x=start_idx, color='white', linewidth=2, alpha=0.8)
            ax2.axvline(x=end_idx, color='white', linewidth=2, alpha=0.8)
    
    # Add time annotations for temporal visualization
    ax2.set_xticks(frame_points)
    ax2.set_xticklabels([f'{t:.0f}s' for t in time_points])
    ax2.set_yticks([])  # No y-axis ticks for 1D visualization
    
    ax2.set_title('Temporal Section Sequence')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Section Type')
    
    # Add legend with sequential numbering
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(i/len(valid_clusters)), 
                           label=f'Section {i+1} (ID: {cluster_id})') 
                      for i, cluster_id in enumerate(valid_clusters)]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Print section information for debugging
    print(f"\nSection Details:")
    for i, cluster_id in enumerate(valid_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        start_time = cluster_indices[0] * hop_length / sr
        end_time = cluster_indices[-1] * hop_length / sr
        length = end_time - start_time
        print(f"  Section {i+1} (ID: {cluster_id}): {start_time:.1f}s - {end_time:.1f}s (length: {length:.1f}s, frames: {len(cluster_indices)})")
        
        # Add text annotations for section boundaries
        if len(cluster_indices) > 1:
            # Add start time label
            ax2.text(start_idx, 0.5, f'{start_time:.0f}s', 
                    ha='right', va='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            # Add end time label
            ax2.text(end_idx, 0.5, f'{end_time:.0f}s', 
                    ha='left', va='center', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Similarity analysis plot saved to: {save_path}")
    
    plt.show()

def analyze_self_similarity(chromagram, sr, hop_length, save_path=None):
    """
    Complete self-similarity analysis with hierarchical clustering.
    
    Args:
        chromagram (np.ndarray): CENS chromagram
        sr (int): Sample rate
        hop_length (int): Hop length
        save_path (str): Path to save visualization
    
    Returns:
        tuple: (sections_array, similarity_matrix, clusters)
    """
    print("="*60)
    print("SELF-SIMILARITY ANALYSIS")
    print("="*60)
    
    # 1. Compute cosine similarity matrix
    print("Computing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity_matrix(chromagram)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # 2. Find natural gap threshold
    print("Finding natural gap threshold...")
    threshold = find_natural_gap_threshold(similarity_matrix)
    
    # 3. Hierarchical clustering with silhouette optimization
    print("Performing hierarchical clustering...")
    clusters, cutoff, silhouette_score_val = hierarchical_clustering_with_silhouette(similarity_matrix)
    
    # 4. Extract section features
    print("Extracting section features...")
    sections_array = extract_section_features(similarity_matrix, clusters, sr, hop_length)
    
    # 5. Create visualization
    print("Creating visualization...")
    plot_similarity_matrix_with_sections(similarity_matrix, clusters, sr, hop_length, save_path)
    
    # 6. Print results
    print(f"\n Analysis complete!")
    print(f" Detected {len(sections_array)} different section types")
    print(f" Silhouette score: {silhouette_score_val:.3f}")
    print(f" Features array shape: {sections_array.shape}")
    
    print(f"\n Section Summary:")
    for i, section in enumerate(sections_array):
        start_time, end_time, length, position, similarity, section_id, frequency, avg_similarity = section
        print(f"   Section {int(section_id)}: {start_time:.1f}s - {end_time:.1f}s "
              f"(length: {length:.1f}s, position: {position:.2f}, "
              f"frequency: {int(frequency)}, similarity: {similarity:.3f})")
    
    return sections_array, similarity_matrix, clusters

if __name__ == "__main__":
    # Example usage
    from ml_chromagram import generate_ml_features
    
    # Generate CENS chromagram
    file_path = "Abba - Dancing Queen.wav"
    feature_matrix, features, analysis = generate_ml_features(file_path, show_plot=False)
    
    

    if feature_matrix is not None:
        # Extract CENS chromagram (first 12 features)
        cens_chroma = features['cens_chroma']
        
        # Analyze self-similarity
        sections_array, similarity_matrix, clusters = analyze_self_similarity(
            cens_chroma, sr=22050, hop_length=512
        )
        
        print(f"\n Ready for ML training!")
        print(f"   Use 'sections_array' for your ML model input")
        print(f"   Features: [start_time, end_time, length, position, similarity, section_id, frequency, avg_similarity]")
    else:
        print("Failed to load audio file") 
    
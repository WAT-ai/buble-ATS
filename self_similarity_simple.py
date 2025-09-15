import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from pathlib import Path
import seaborn as sns

'''
def self_similarity_simple(feature_matrix):
    #12 Notes, sampled 9967 times
'''

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


def plot_similarity_matrix(similarity_matrix, title="Self-Similarity Matrix", cmap="viridis", save_path=None):
    """
    Plot the similarity matrix as a heatmap.

    Args:
        similarity_matrix (np.ndarray): Square matrix of similarities
        title (str): Plot title
        cmap (str): Colormap for the heatmap
        save_path (str|Path|None): If provided, save the figure to this path
    """
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        similarity_matrix,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        cbar=True,
        xticklabels=False, #goes from 0 to 9966 (L to R)
        yticklabels=False, #goes from 0 to 9966 (T to B)
    )
    ax.invert_yaxis()  # Flip y-axis so 0 is at bottom, 9966 at top
    ax.set_title(title)
    ax.set_xlabel("Time Frame (L to R)")
    ax.set_ylabel("Time Frame (B to T)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()








if __name__ == "__main__":
    # Example usage
    from ml_chromagram import generate_ml_features
    
    # Generate CENS chromagram
    file_path = "Abba - Dancing Queen.wav"
    feature_matrix, features, analysis = generate_ml_features(file_path, show_plot=False)

    similarity_matrix = compute_cosine_similarity_matrix(feature_matrix)
    print(similarity_matrix)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Plot similarity matrix
    title = f"Self-Similarity: {Path(file_path).name}"
    plot_similarity_matrix(similarity_matrix, title=title, cmap="magma")


    #self_similarity_simple(feature_matrix)
    
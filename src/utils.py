from sklearn.metrics import hamming_loss as skl_hamming_loss

def calculate_hamming_loss(y_true, y_pred):
    """
    Calculate the Hamming Loss for multi-label classification.
    y_true and y_pred should be numpy arrays or tensors of shape (num_samples, num_labels)
    """
    return skl_hamming_loss(y_true, y_pred)

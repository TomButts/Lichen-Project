import scipy.io as sio

def parse(targets_path):
    """Reads 102flowers labels into a list

    Args:
        targets_path: The path to the matlab file

    Returns:
        targets: A list of target labels
    """
    targets = sio.loadmat(targets_path)

    return targets['labels'][0]

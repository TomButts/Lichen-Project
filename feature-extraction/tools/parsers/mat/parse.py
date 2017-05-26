import scipy.io as sio

def parse(targets_path):
    """Parse a matlab labels file

    Args:
        targets_path: The path to the matlab file

    Returns:
        targets: A list of target labels
    """
    targets = sio.loadmat(targets_path)

    return targets['labels'][0]

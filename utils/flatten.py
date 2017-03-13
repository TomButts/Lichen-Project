
def flatten(container):
    """Flattens an array or dict

    Args:
        container: multi dimensional array of arbitrary nest level

    Returns:
        A 1D list
    """
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

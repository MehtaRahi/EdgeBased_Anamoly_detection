from src.data.smd_loader import load_smd
from src.data.skab_loader import load_skab
from src.data.smap_loader import load_smap
from src.data.nab_loader import load_nab


def load_data(machine_id=None, dataset="smd", **kwargs):

    if dataset == "smd":
        return load_smd(machine_id, **kwargs)

    elif dataset == "skab":
        X, y = load_skab(**kwargs)

        split = int(0.7 * len(X))

        return X[:split], X[split:], y[split:]

    elif dataset == "smap":
        return load_smap(**kwargs)

    elif dataset == "nab":
        return load_nab(**kwargs)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
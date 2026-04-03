from src.data.smd_loader import load_smd


def load_data(machine_id, dataset="smd", **kwargs):

    if dataset == "smd":
        return load_smd(machine_id, **kwargs)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
import torch

def load_simmim_weights(backbone, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if "model" in ckpt:
        ckpt = ckpt["model"]

    cleaned = {}
    state = backbone.state_dict()

    for k, v in ckpt.items():
        k2 = k
        for prefix in ["module.", "encoder.", "backbone.", "model."]:
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]

        if k2 in state and state[k2].shape == v.shape:
            cleaned[k2] = v

    msg = backbone.load_state_dict(cleaned, strict=False)

    print("\n=== SimMIM SSL Load Report ===")
    print("Loaded:", len(cleaned))
    print("Missing:", len(msg.missing_keys))
    print("Unexpected:", len(msg.unexpected_keys))
    print("=================================\n")

    return backbone

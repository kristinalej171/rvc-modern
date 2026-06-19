import torch


def get_rmvpe(model_path="assets/rmvpe/rmvpe.pt", device=torch.device("cpu")):
    from infer.lib.rmvpe import E2E

    model = E2E(4, 1, (2, 2))
    # 🔒 SECURITY: weights_only=True — RMVPE checkpoint это state_dict (dict of Tensors)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model
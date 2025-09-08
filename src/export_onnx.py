import torch
import numpy as np
from .network import PolicyValueNet

def export(ckpt="checkpoints/az_v0_final.pt", C=18, onnx_path="az_chess.onnx"):
    device = "cpu"
    model = PolicyValueNet(in_channels=C).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    dummy = torch.randn(1, C, 8, 8, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["board"], output_names=["policy_logits", "value"],
        dynamic_axes={"board": {0:"B"}, "policy_logits": {0:"B"}, "value": {0:"B"}},
        opset_version=17
    )
    print("Exported:", onnx_path)
    print("Note: To optimize the model, you can optionally install onnxsim and run:")
    print(f"python -m onnxsim {onnx_path} {onnx_path.replace('.onnx', '_sim.onnx')}")

if __name__ == "__main__":
    export()

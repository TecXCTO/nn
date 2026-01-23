# Deployment (ONNX / TorchScript)
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")

# Or TorchScript
scripted = torch.jit.script(model)
scripted.save("model.pt")

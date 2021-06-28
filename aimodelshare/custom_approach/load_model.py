import onnxruntime as rt
import string

def load_model(filename="runtime_model.onnx"):
    model = rt.InferenceSession('/var/lib/docker/' + filename)
    return model
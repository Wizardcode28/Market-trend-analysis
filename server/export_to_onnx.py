# export_to_onnx.py
import os, joblib, torch, numpy as np

# Re-declare model class EXACTLY as training
import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out).squeeze(-1)

# Training defaults (from your train script)
FALLBACK_FEATURES = ['open','high','low','close','adjclose','volume','rsiadjclose15','rsivolume15','rsiad']
FALLBACK_LOOKBACK = 30
FALLBACK_MODEL_KW = {"input_dim": len(FALLBACK_FEATURES), "hidden_dim":64, "num_layers":2, "dropout":0.3}

MODEL_PTH = "models/model.pth"
BUNDLE_PKL = "models/bundle.pkl"
OUT_ONNX = "models/model.onnx"
OPSET = 17

def load_bundle_fallback(bundle_path):
    if not os.path.exists(bundle_path):
        print("bundle not found, falling back to defaults")
        return {"scaler": None, "features": FALLBACK_FEATURES, "lookback": FALLBACK_LOOKBACK, "model_kwargs": FALLBACK_MODEL_KW}
    raw = joblib.load(bundle_path)
    print("bundle type:", type(raw))
    if isinstance(raw, dict):
        bundle = raw.copy()
        # ensure features & lookback exist
        if "features" not in bundle:
            bundle["features"] = FALLBACK_FEATURES
        if "lookback" not in bundle:
            bundle["lookback"] = FALLBACK_LOOKBACK
        if "model_kwargs" not in bundle:
            # try to infer input_dim
            bundle["model_kwargs"] = bundle.get("model_kwargs", {"input_dim": len(bundle["features"]), "hidden_dim":64, "num_layers":2, "dropout":0.3})
        # normalize features to strings
        bundle["features"] = [str(f).strip().lower() for f in bundle["features"]]
        return bundle
    else:
        # bundle is probably a scaler instance (your case). Use fallback metadata but keep scaler
        if hasattr(raw, "transform"):
            print("bundle is scaler instance; using fallback metadata")
            bundle = {"scaler": raw, "features": FALLBACK_FEATURES, "lookback": FALLBACK_LOOKBACK, "model_kwargs": FALLBACK_MODEL_KW}
            return bundle
        # list/tuple heuristic
        if isinstance(raw, (list, tuple)):
            # try to find scaler and features
            scaler = None; features = None; lookback = FALLBACK_LOOKBACK; mk = FALLBACK_MODEL_KW
            for it in raw:
                if hasattr(it, "transform"):
                    scaler = it
                if isinstance(it, (list, tuple)) and all(isinstance(x, str) for x in it):
                    features = list(it)
                if isinstance(it, dict):
                    mk = it.get("model_kwargs", mk)
                    lookback = it.get("lookback", lookback)
            if scaler is not None and features is not None:
                bundle = {"scaler": scaler, "features": [str(f).lower() for f in features], "lookback": lookback, "model_kwargs": mk}
                return bundle
        raise RuntimeError("Unsupported bundle.pkl type: %s" % type(raw))

def export():
    bundle = load_bundle_fallback(BUNDLE_PKL)
    features = bundle["features"]
    lookback = int(bundle["lookback"])
    model_kwargs = bundle.get("model_kwargs", FALLBACK_MODEL_KW)
    input_dim = int(model_kwargs.get("input_dim", len(features)))

    print("Using features:", features)
    print("Using lookback:", lookback)
    print("Model kwargs:", model_kwargs)

    if not os.path.exists(MODEL_PTH):
        raise FileNotFoundError(MODEL_PTH + " not found")

    state = torch.load(MODEL_PTH, map_location="cpu")
    print("Loaded model.pth type:", type(state))

    # instantiate model
    hidden_dim = int(model_kwargs.get("hidden_dim", 64))
    num_layers = int(model_kwargs.get("num_layers", 1))
    dropout = float(model_kwargs.get("dropout", 0.0))
    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    # load state whether state is dict or tuple
    if isinstance(state, dict):
        # try direct
        try:
            model.load_state_dict(state)
            print("Loaded state_dict into model.")
        except Exception:
            # common wrappers
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                print("Loaded model_state_dict key.")
            elif "state_dict" in state:
                model.load_state_dict(state["state_dict"])
                print("Loaded state_dict key.")
            else:
                # sometimes state is {'model': state_dict}
                for k,v in state.items():
                    if isinstance(v, dict):
                        try:
                            model.load_state_dict(v)
                            print("Loaded nested dict state under key:", k)
                            break
                        except Exception:
                            pass
    elif isinstance(state, (list, tuple)) and isinstance(state[0], dict):
        model.load_state_dict(state[0])
        print("Loaded tuple(state_dict,...)")
    elif isinstance(state, nn.Module):
        model = state
        print("Loaded full model object.")
    else:
        raise RuntimeError("Unrecognized model.pth contents")

    model.eval()
    dummy = torch.randn(1, lookback, input_dim, dtype=torch.float32)

    print("Exporting ONNX to:", OUT_ONNX)
    torch.onnx.export(model, dummy, OUT_ONNX,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=OPSET)
    print("✅ Exported", OUT_ONNX)

if __name__ == "__main__":
    export()

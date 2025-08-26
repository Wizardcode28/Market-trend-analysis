import joblib, pprint, os
p = "models/bundle.pkl"
if not os.path.exists(p):
    print("bundle.pkl not found at", p)
else:
    obj = joblib.load(p)
    print("bundle type:", type(obj))
    if isinstance(obj, dict):
        print("bundle keys:", list(obj.keys()))
        if "features" in obj:
            print("features type:", type(obj["features"]))
            print("sample features (first 10):", obj["features"][:10])
        if "model_kwargs" in obj:
            print("model_kwargs:", obj["model_kwargs"])
        if "scaler" in obj:
            print("scaler type:", type(obj["scaler"]))
    else:
        import collections
        if isinstance(obj, (list, tuple)):
            print("bundle is list/tuple, length:", len(obj))
            for i, it in enumerate(obj):
                print(i, type(it), getattr(it, "__class__", None))
                # If small, print repr
                if i < 5:
                    print(" repr:", repr(it)[:200])
        else:
            print("bundle repr (first 400 chars):", repr(obj)[:400])

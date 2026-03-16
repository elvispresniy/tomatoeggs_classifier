from ultralytics import FastSAM

_model = None

def get_fastsam_model(model_name="FastSAM-s.pt"):
    global _model
    if _model is None:
        print("Loading FastSAM model...")
        _model = FastSAM(model_name)
    return _model
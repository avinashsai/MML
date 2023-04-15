import torch
from torch.optim import AdamW

from transformers import (
    AutoModelForSequenceClassification
)

from models.fit import get_fit_model
from models.blip import get_blip_model
from models.albef import get_albef_model
from models.alpro import get_alpro_model
from models.meter import get_meter_model
from models.violet import get_violet_model
from models.clip import get_clip_model

def load_model(modelpath, modelname, numclasses):
    if('fit' in modelname):
        model = get_fit_model(numclasses, modelpath)
        print("######################################################################")
        print("Loaded Frozen in time model")
    elif('blip' in modelname):
        model = get_blip_model(numclasses, modelpath)
        print("######################################################################")
        print("Loaded BLIP model")
    elif('albef' in modelname):
        model = get_albef_model(numclasses, modelpath)
        print("######################################################################")
        print("Loaded ALBEF model")
    elif('alpro' in modelname):
        model = get_alpro_model(numclasses, modelpath)
        print("######################################################################")
        print("Loaded ALPRO model")
    elif('meter' in modelname):
        model = get_meter_model(numclasses, modelpath)
        print("######################################################################")
        print("Loaded METER model")
    elif('violet' in modelname):
        model = get_violet_model(numclasses, modelpath)
        print("######################################################################")
        print("Loaded VIOLET model")
    elif('clip' in modelname):
        model = get_clip_model(numclasses, modelpath)
        print("######################################################################")
        print("Loaded CLIP model")
        
    return model

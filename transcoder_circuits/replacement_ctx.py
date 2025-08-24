# --- context manager for replacing MLP sublayers with transcoders ---
import torch

class TranscoderWrapper(torch.nn.Module):
    def __init__(self, transcoder, layer_idx=-1):
        super().__init__()
        self.transcoder = transcoder
        self.cache = False
        self.layer_idx=layer_idx
        self.act_cache = {}
    
    def forward(self, x):
        result = self.transcoder(x)[0].to(torch.bfloat16)
        if self.cache:
            self.act_cache[f'mlp_input.{self.layer_idx}'] = x
            self.act_cache[f'mlp_output.{self.layer_idx}'] = result
        return result, None

class TranscoderReplacementContext:
    def __init__(self, model, transcoders):
        self.layers = [t.cfg.hook_point_layer for t in transcoders]
        self.original_mlps = [ model.model.layers[i].mlp for i in self.layers ]
        
        self.transcoders = transcoders
        #self.layers = layers
        self.model = model
    
    def __enter__(self):
        for transcoder in self.transcoders:
           self.model.model.layers[transcoder.cfg.hook_point_layer].mlp = TranscoderWrapper(transcoder, layer_idx=transcoder.cfg.hook_point_layer)

    def __exit__(self, exc_type, exc_value, exc_tb):
        for layer, mlp in zip(self.layers, self.original_mlps):
            self.model.model.layers[layer].mlp = mlp

class ZeroAblationWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return (x*0.0).to(torch.bfloat16), None

class ZeroAblationContext:
    def __init__(self, model, layers):
        self.original_mlps = [ model.model.layers[i].mlp for i in layers ]
        
        self.layers = layers
        self.model = model
    
    def __enter__(self):
        for layer in self.layers:
           self.model.model.layers[layer].mlp = ZeroAblationWrapper()

    def __exit__(self, exc_type, exc_value, exc_tb):
        for layer, mlp in zip(self.layers, self.original_mlps):
            self.model.model.layers[layer].mlp = mlp
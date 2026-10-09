"""Inference adapter for an independently obtained official SmoothNet checkpoint."""
from __future__ import annotations


# File identity
import hashlib
def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# Model and inference
from pathlib import Path
import importlib.util
import numpy as np

PAPER_CHECKPOINT_SHA256 = '3bfe43db4ee320f277d792b2252a4f4bb10ddefd46c023ff61d3781be7f42a58'


class SmoothNet32:
    """Use the supplied official checkout and PW3D--SPIN checkpoint.

    Both final-paper correction experiments use this checkpoint. Supply the
    original checkout under its own Apache-2.0 license. This class does not
    download, train, choose, or silently replace model weights.
    """
    def __init__(self, repository, checkpoint, device='cpu', verify_paper_hash=True):
        import torch
        source=Path(repository)/'lib/models/smoothnet.py'
        if not source.is_file():
            raise FileNotFoundError('Expected official lib/models/smoothnet.py')
        digest=file_hash(checkpoint)
        if verify_paper_hash and digest!=PAPER_CHECKPOINT_SHA256:
            raise ValueError('Checkpoint differs from the frozen paper model')
        # This checkpoint is authenticated by its exact hash before allowing
        # historical torch serialization. Other models require weights-only.
        archive=torch.load(checkpoint,map_location='cpu',
                           weights_only=(digest!=PAPER_CHECKPOINT_SHA256))
        state={key.removeprefix('module.'):value for key,value in archive['state_dict'].items()}
        spec=importlib.util.spec_from_file_location('cmat_external_smoothnet',source)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hidden=state['encoder.0.weight'].shape[0]
        residual=state['res_blocks.0.linear1.weight'].shape[0]
        blocks=max(int(k.split('.')[1]) for k in state if k.startswith('res_blocks.'))+1
        self.model=module.SmoothNet(32,32,hidden,residual,blocks).to(device).eval()
        self.model.load_state_dict(state,strict=True)
        self.torch,self.device=torch,device
        training_source = 'PW3D-SPIN' if digest == PAPER_CHECKPOINT_SHA256 else 'unverified/custom'
        self.metadata=dict(checkpoint_sha256=digest,training_source=training_source,
                           window=32,stride=1,hidden=hidden,residual_hidden=residual,
                           blocks=blocks,training_or_finetuning=False)

    def predict(self, observed):
        observed=np.asarray(observed)
        if observed.ndim!=3 or observed.shape[-1]!=3 or len(observed)==0:
            raise ValueError('Expected nonempty [T,J,3] positions')
        if not np.isfinite(observed).all():
            raise ValueError('Positions must be finite')
        frames,joints,_=observed.shape
        padded=np.pad(observed,((0,max(0,32-frames)),(0,0),(0,0)))
        total=np.zeros_like(padded,dtype=np.float64)
        count=np.zeros(len(padded))
        starts=np.arange(len(padded)-32+1)
        with self.torch.inference_mode():
            for start in range(0,len(starts),256):
                indices=starts[start:start+256,None]+np.arange(32)[None]
                tensor=self.torch.as_tensor(
                    padded[indices].reshape(len(indices),32,-1).transpose(0,2,1),
                    device=self.device,dtype=self.torch.float32)
                output=self.model(tensor).cpu().numpy().transpose(0,2,1)
                output=output.reshape(len(indices),32,joints,3)
                np.add.at(total,indices.ravel(),output.reshape(-1,joints,3))
                np.add.at(count,indices.ravel(),1)
        return (total/count[:,None,None])[:frames].astype(np.float32)

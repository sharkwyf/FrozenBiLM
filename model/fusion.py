import torch
import torch.nn as nn

class InputFusionWrapperModel(nn.Module):
    """
    An input fusion model to wrap video features with idm features
    """
    def __init__(self, config, model):
        super().__init__()
        self._use_idm_features = config.use_idm_features
        self.model = model
        self.proj = nn.Linear(config.idm_features_dim, config.features_dim, bias=True)

    def forward(self, video, video_mask, input_ids, attention_mask, labels, idm_feats=None, *arg, **kwargs):
        if idm_feats is not None:
            x = self.proj(idm_feats)
            assert video.shape == x.shape, (video.shape, x.shape)
            video += x
        output = self.model(
            video=video,
            video_mask=video_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            *arg,
            **kwargs,
        )
        return output

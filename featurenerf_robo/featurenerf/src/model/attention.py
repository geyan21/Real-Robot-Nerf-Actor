import torch
import torch.nn as nn


class LanguageInformedVisualAttention(nn.Module):
    """
    gated cross attention between vision and language
    """
    def __init__(self, language_dim=1024, visual_dim=512):
        super().__init__()
        self.lang_dim = language_dim
        self.visual_dim = visual_dim
        self.projection = nn.Linear(self.lang_dim, self.visual_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.ReLU(),
            nn.Linear(self.visual_dim, self.visual_dim)
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

        self.attention = nn.MultiheadAttention(self.visual_dim, 1)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))


    def forward(self, visual_embed, language_embed):
        language_embed = self.projection(language_embed)
        
        # 1. Gated Cross Attention
        language_embed = language_embed.unsqueeze(1).repeat(1, visual_embed.size(1), 1)
        visual_embed = visual_embed + self.attn_gate.tanh() * self.attention(query=visual_embed, key=language_embed, value=language_embed)[0]
    
        # 2. Gated Feed Forward
        visual_embed = visual_embed + self.ff_gate.tanh() * self.ff(visual_embed)

        # print("self.attn_gate: ", self.attn_gate)
        # print("self.ff_gate: ", self.ff_gate)

        return visual_embed
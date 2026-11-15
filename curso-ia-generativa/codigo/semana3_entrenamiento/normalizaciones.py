"""
BatchNorm, LayerNorm y RMSNorm escritas desde cero y verificadas contra torch.nn.

  BatchNorm : normaliza cada FEATURE usando estadísticas del BATCH   (dim 0)  -> CNN / MLP
  LayerNorm : normaliza cada MUESTRA usando sus propias features     (dim -1) -> Transformers
  RMSNorm   : como LayerNorm pero sin restar la media: x / rms(x) * g  -> LLaMA, Mistral, Qwen...

    python normalizaciones.py
"""
import torch
import torch.nn as nn


class BatchNorm1dScratch(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma, self.beta = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.register_buffer("run_mean", torch.zeros(d))
        self.register_buffer("run_var", torch.ones(d))

    def forward(self, x):                       # x: [B, D]
        if self.training:
            mu, var = x.mean(0), x.var(0, unbiased=False)
            with torch.no_grad():
                self.run_mean.lerp_(mu, self.momentum)
                self.run_var.lerp_(x.var(0, unbiased=True), self.momentum)
        else:
            mu, var = self.run_mean, self.run_var
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta


class LayerNormScratch(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma, self.beta = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))

    def forward(self, x):                       # x: [..., D]
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):                       # x: [..., D]
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.g * x / rms


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(32, 64) * 3 + 5
    for nombre, mio, ref in [
        ("BatchNorm", BatchNorm1dScratch(64), nn.BatchNorm1d(64)),
        ("LayerNorm", LayerNormScratch(64), nn.LayerNorm(64)),
    ]:
        mio.train(), ref.train()
        d = (mio(x) - ref(x)).abs().max().item()
        print(f"{nombre:<10} máx |mío - torch.nn| = {d:.2e}")
    # chequeo de BatchNorm en modo eval (usa estadísticas acumuladas)
    bn_m, bn_r = BatchNorm1dScratch(64), nn.BatchNorm1d(64)
    for _ in range(50):
        xb = torch.randn(32, 64) * 3 + 5
        bn_m.train(), bn_r.train(); bn_m(xb), bn_r(xb)
    bn_m.eval(), bn_r.eval()
    print(f"BatchNorm(eval) máx |mío - torch.nn| = {(bn_m(x)-bn_r(x)).abs().max().item():.2e}")

    r = RMSNorm(64)
    y = r(x)
    print(f"RMSNorm: rms de la salida por muestra ≈ {y.pow(2).mean(-1).sqrt().mean():.4f} (debe ser ~1)")
    print(f"LayerNorm: media≈{LayerNormScratch(64)(x).mean():.1e}, std≈{LayerNormScratch(64)(x).std(-1).mean():.3f}")
    if hasattr(nn, "RMSNorm"):
        d = (nn.RMSNorm(64, eps=1e-6)(x) - r(x)).abs().max().item()
        print(f"RMSNorm    máx |mío - torch.nn.RMSNorm| = {d:.2e}")

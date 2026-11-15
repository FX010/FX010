"""
Mapeo tensorial (shape tracking): ver cómo cambian las dimensiones capa por capa.

Muestra tres técnicas, de más manual a más automática:
  A. Anotación dimensional en comentarios (la práctica que más bugs evita).
  B. Hooks de PyTorch que imprimen la forma de entrada/salida de cada módulo.
  C. Reshape de multi-head attention paso a paso: [B,T,C] -> [B,H,T,Dh] -> [B,T,C]

(Opcionales para tu máquina: `pip install torchinfo einops torchviz` y usa
 torchinfo.summary(modelo, input_size=(B,...)) y einops.rearrange.)

    python shape_tracking.py
"""
import torch
import torch.nn as nn


def registrar_formas(modelo: nn.Module):
    """Engancha un forward-hook a cada hoja del modelo y devuelve los handles."""
    def hook(mod, entrada, salida):
        e = tuple(entrada[0].shape) if isinstance(entrada, tuple) and torch.is_tensor(entrada[0]) else "?"
        s = tuple(salida.shape) if torch.is_tensor(salida) else "?"
        n_par = sum(p.numel() for p in mod.parameters(recurse=False))
        print(f"   {mod.__class__.__name__:<12} {str(e):<22} -> {str(s):<22} params={n_par}")

    return [m.register_forward_hook(hook) for m in modelo.modules() if len(list(m.children())) == 0]


print("B. Hooks sobre una CNN pequeña")
cnn = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # [B,1,16,16] -> [B,8,16,16] -> [B,8,8,8]
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # -> [B,16,8,8] -> [B,16,4,4]
    nn.Flatten(), nn.Linear(16 * 4 * 4, 4),                        # -> [B,256] -> [B,4]
)
hs = registrar_formas(cnn)
cnn(torch.randn(2, 1, 16, 16))
for h in hs:
    h.remove()

print("\nC. Reshape de Multi-Head Attention (B=2, T=5, C=32, H=4 -> Dh=8)")
B, T, C, H = 2, 5, 32, 4
Dh = C // H
x = torch.randn(B, T, C)
qkv = nn.Linear(C, 3 * C)(x)                    # [B,T,3C]
q, k, v = qkv.split(C, dim=-1)                  # 3 x [B,T,C]
print(f"   q            {tuple(q.shape)}")
q = q.view(B, T, H, Dh).transpose(1, 2)         # [B,H,T,Dh]  (cada cabeza ve un subespacio de Dh dims)
k = k.view(B, T, H, Dh).transpose(1, 2)
v = v.view(B, T, H, Dh).transpose(1, 2)
print(f"   q por cabeza {tuple(q.shape)}")
att = (q @ k.transpose(-2, -1)) / Dh ** 0.5     # [B,H,T,Dh] @ [B,H,Dh,T] -> [B,H,T,T]
print(f"   scores       {tuple(att.shape)}   (cada cabeza: T x T relaciones token-token)")
att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf")).softmax(-1)
out = att @ v                                   # [B,H,T,T] @ [B,H,T,Dh] -> [B,H,T,Dh]
out = out.transpose(1, 2).contiguous().view(B, T, C)   # concatenar cabezas -> [B,T,C]
print(f"   salida       {tuple(out.shape)}")

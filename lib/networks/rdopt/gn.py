import torch
import torch.nn as nn


@torch.jit.script
def computeJtJandJte(x, J, e, lam):
    Jt = torch.transpose(J, dim0=1, dim1=2)
    JtJ = torch.bmm(Jt, J)
    Jte = torch.bmm(Jt, e)
    identity = torch.eye(JtJ.shape[1], device=Jt.device).unsqueeze(0)
    lamD = lam * identity
    JtJ = JtJ + lamD

    return JtJ, Jte


class GNLayer(nn.Module):
    def __init__(self, out_channels):
        super(GNLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(out_channels, 16), nn.ReLU(),
                                nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1),
                                nn.ReLU())

    def forward(self, x, e, J):
        bs = x.shape[0]
        e_mean = e.mean(dim=1)
        lam = self.fc(e_mean)[:, None]

        J = J.reshape(bs, -1, 6)
        e = e.reshape(bs, -1, 1)

        JtJ, Jte = computeJtJandJte(x, J, e, lam)

        try:
            delta_x = torch.linalg.solve(JtJ, Jte)[:, :, 0]
            x = x + delta_x
        except RuntimeError:
            pass

        return x

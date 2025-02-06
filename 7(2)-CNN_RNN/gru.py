import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.W_z = nn.Linear(input_size, hidden_size, bias=True)
        self.W_r = nn.Linear(input_size, hidden_size, bias=True)
        self.W_h = nn.Linear(input_size, hidden_size, bias=True)

        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)
        
    
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h))
        h_next = (1 - z) * h + z * h_tilde
        
        return h_next


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        batch_size, seq_len, _ = inputs.shape
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        outputs_list = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]            
            h = self.cell(x_t, h)            
            outputs_list.append(h.unsqueeze(1))   

        outputs = torch.cat(outputs_list, dim=1)
        return outputs
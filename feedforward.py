import	torch
import	torch.nn			as	nn
import	torch.nn.functional	as	F

class feedforward_block(nn.Module):
	def __init__(self, in_out_size: int, hidden_size: int, device: str = "cpu") -> None:
		super().__init__()
		self.layer1: nn.Linear = nn.Linear(
			in_features = in_out_size, 
			out_features = hidden_size,
			bias = True,
			device = device
		)
		self.activate: nn.ReLU = nn.ReLU()
		self.layer2: nn.Linear = nn.Linear(
			in_features = hidden_size,
			out_features = in_out_size,
			bias = True,
			device = device
		)
	def forward(self, x: torch.Tensor):
		expand: torch.Tensor = self.activate(self.layer1(x))
		residual: torch.Tensor = self.layer2(expand)
		return x + residual

 
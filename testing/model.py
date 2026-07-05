import	torch
import	torch.nn			as		nn
import	torch.optim			as		optim
import	torch.nn.functional	as		F
from	torch.utils.data	import	DataLoader

class strongly_connected_arch(nn.Module):
	def __init__(self, device: str = "cpu") -> None:
		super().__init__()
		self.layer1: nn.Linear = nn.Linear(in_features = 28 * 28, out_features = 100, bias = True, device = device)
		self.layer2: nn.Linear = nn.Linear(in_features = 100, out_features = 50, bias = True, device = device)
		self.layer3: nn.Linear = nn.Linear(in_features = 50, out_features = 10, bias = True, device = device)
	def forward(self, input_image: torch.Tensor) -> torch.Tensor:
		layer1_output: torch.Tensor = nn.ReLU(self.layer1(input_image))
		layer2_output: torch.Tensor = nn.ReLU(self.layer2(layer1_output))
		layer3_output: torch.Tensor = nn.ReLU(self.layer3(layer2_output))
		return nn.Softmax(layer3_output)
	
class cnn_arch(nn.Module):
	def __init__(self, device: str = "cpu") -> None:
		super().__init__()
		self.layer1_features: nn.Conv2d = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 7, stride = 2, device = device)
		self.layer2_features: nn.Conv2d = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 5, stride = 3, device = device)
		self.layer3_features: nn.Conv2d = nn.Conv2d(in_channels = 20, out_channels = 64, kernel_size = 3, stride = 1, device = device)
		self.mlp_layer1: nn.LazyLinear = nn.LazyLinear(out_features = 50, bias = True, device = device)
		self.mlp_layer2: nn.Linear = nn.Linear(in_features = 50, out_features = 20, bias = True, device = device)
		self.mlp_layer3: nn.Linear = nn.Linear(in_features = 20, out_features = 10, bias = True, device = device)
	def forward(self, input_image: torch.Tensor) -> torch.Tensor:
		featurised_1: torch.Tensor = self.layer1_features(input_image)
		featurised_2: torch.Tensor = self.layer2_features(featurised_1)
		featurised_3: torch.Tensor = self.layer3_features(featurised_2)
		featurised_3 = featurised_3.reshape((-1,))
		layer1_inference: torch.Tensor = nn.ReLU(self.layer1(featurised_3))
		layer2_inference: torch.Tensor = nn.ReLU(self.layer2(layer1_inference))
		layer3_inference: torch.Tensor = nn.ReLU(self.layer3(layer2_inference))
		return nn.Softmax(layer3_inference)

class full_model:
	def __init__(self, dataloader: DataLoader, lr: float = 0.001, momentum: float = 0.0, model_type: str = "mlp", device: str = "cpu") -> None:
		self.model: nn.Module = (strongly_connected_arch if model_type == "nlp" else cnn_arch)(device).to(device)
		self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
		self.optimiser: optim.Optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum = momentum)
		self.dataload: DataLoader = dataloader
		self.device: str = device
	def train(self) -> tuple[list[tuple[float, int]], int]:
		size: int = len(self.dataload.dataset)
		loss_history: list[tuple[float, int]] = []
		self.model.train()
		for batch, (X, y) in enumerate(self.dataload):
			X, y = X.to(self.device), y.to(self.device)
			pred: torch.Tensor = self.model(X)
			loss: torch.Tensor = self.loss_fn(pred, y)
			loss.backward()
			self.optimiser.step()
			self.optimiser.zero_grad()
			if batch % 100 == 0:
				loss_history.append([loss.item(), (batch + 1) * len(X)])
		return [loss_history, size]
	def test(self) -> tuple[float, float]:
		size: int = len(self.dataload.dataset)
		num_batches: int = len(self.dataload)
		self.model.eval()
		test_loss: float = 0
		correct: int = 0
		with torch.no_grad():
			for X, y in self.dataload:
				X, y = X.to(self.device), y.to(self.device)
				pred: torch.Tensor = self.model(X)
				test_loss += self.loss_fn(pred, y).item()
				correct += (pred.argmax(1) == y).type(torch.float).sum().item()
		return [test_loss / num_batches, correct / size]
	def save(self, filename: str) -> None:
		torch.save(self.model.state_dict(), filename)
		


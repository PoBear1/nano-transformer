import	torch
import	torch.nn 				as		nn
import	torch.optim				as		optim
import	torch.nn.functional		as		F
from	torch.utils.data		import	DataLoader

import	torchvision
from	torchvision.transforms	import	v2

import model					as		M

batch_size: int = 64
epochs: int = 5
MNIST_data: tuple[DataLoader, DataLoader] = (
	torchvision.datasets.MNIST(
		root = "data", 
		train = True,
		download = True,
		transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
	),
	torchvision.datasets.MNIST(
		root = "data", 
		train = False,
		download = True,
		transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)])
	)
)

MNIST_dataloader: tuple[DataLoader, DataLoader] = (
	DataLoader(MNIST_data[0], batch_size = batch_size),
	DataLoader(MNIST_data[1], batch_size = batch_size)
)

model: M.full_model = M.full_model(MNIST_dataloader[0], 0.001, 0.0, model_type = "mlp")
for t in range(epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	history, size = model.train()
	for loss, current in history:
		print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
	test_loss, correct = model.test()
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
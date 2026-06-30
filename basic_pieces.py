import	torch
import	numpy				as	np
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
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		expand: torch.Tensor = self.activate(self.layer1(x))
		residual: torch.Tensor = self.layer2(expand)
		return x + residual

class attention(nn.Module):
	def __init__(self, qk_embed_size: int, v_embed_size: int, query_size: int, key_size: int, value_size: int, device: str = "cpu") -> None:
		super().__init__()
		self.query_embed: nn.Linear = nn.Linear(
			in_features = query_size, 
			out_features = qk_embed_size, 
			bias = False,
			device = device
		)
		self.key_embed: nn.Linear = nn.Linear(
			in_features = key_size,
			out_features = qk_embed_size,
			bias = False,
			device = device
		)
		self.value_embed: nn.Linear = nn.Linear(
			in_features = value_size,
			out_features = v_embed_size,
			bias = False,
			device = device
		)
		self.scale_dk: float = 1 / np.sqrt(qk_embed_size)
		self.softmaxing: nn.Softmax = nn.Softmax(dim = -1)
	def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
		# lemme reason on the final sizes assuming single input
		# query = n x dQK
		q_embed: torch.Tensor = self.query_embed(query)
		# key = n x dQK
		k_embed: torch.Tensor = self.key_embed(key)
		# value = n x dV
		v_embed: torch.Tensor = self.value_embed(value)
		# dot_product = n x n
		dot_product: torch.Tensor = self.scale_dk * torch.matmul(q_embed, torch.transpose(k_embed, 0, 1))
		# softmaxed = n x n
		softmaxed: torch.Tensor = self.softmaxing(dot_product)
		# v_embed = n x dV
		return torch.matmul(softmaxed, v_embed)

class multihead_attention(nn.Module):
	def __init__(self, qk_embed_size: int, v_embed_size: int, query_size: int, key_size: int, value_size: int, output_size: int, num_heads: int, device: str = "cpu"):
		super().__init__()
		self.query_embed: nn.Linear = nn.Linear(
			in_features = query_size, 
			out_features = num_heads * qk_embed_size, 
			bias = False,
			device = device
		)
		self.key_embed: nn.Linear = nn.Linear(
			in_features = key_size,
			out_features = num_heads * qk_embed_size,
			bias = False,
			device = device
		)
		self.value_embed: nn.Linear = nn.Linear(
			in_features = value_size,
			out_features = num_heads * v_embed_size,
			bias = False,
			device = device
		)
		self.scale_dk: float = 1 / np.sqrt(qk_embed_size)
		self.softmaxing: nn.Softmax = nn.Softmax(dim = -1)
		self.num_heads: int = num_heads
		self.qk_size: int = qk_embed_size
		self.v_size: int = v_embed_size
		self.final_project: nn.Linear = nn.Linear(
			in_features = num_heads * v_embed_size,
			out_features = output_size,
			bias = False,
			device = device
		)
		
	def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
		# lemme reason on the final sizes assuming single input
		# query = h x n x dQK
		q_embed: torch.Tensor = torch.transpose(torch.reshape(self.query_embed(query), (-1, self.num_heads, self.qk_size)), 0, 1)
		# key = h x n x dQK
		k_embed: torch.Tensor = torch.transpose(torch.reshape(self.key_embed(key), (-1, self.num_heads, self.qk_size)), 0, 1)
		# value = h x n x dV
		v_embed: torch.Tensor = torch.transpose(torch.reshape(self.value_embed(value), (-1, self.num_heads, self.v_size)), 0, 1)
		# dot_product = h x n x n
		dot_product: torch.Tensor = self.scale_dk * torch.matmul(q_embed, torch.transpose(k_embed, -2, -1))
		# softmaxed = h x n x n
		softmaxed: torch.Tensor = self.softmaxing(dot_product)
		# head_embed = h x n x dV
		head_embedded = torch.matmul(softmaxed, v_embed)
		# final output = n x output_size
		return self.final_project(torch.reshape(torch.transpose(head_embedded, 0, 1), (self.v_size * self.num_heads, -1)))
	
class masked_attention(nn.Module):
	def __init__(self, qk_embed_size: int, v_embed_size: int, query_size: int, key_size: int, value_size: int, output_size: int, num_heads: int, device: str = "cpu"):
		super().__init__()
		self.query_embed: nn.Linear = nn.Linear(
			in_features = query_size, 
			out_features = num_heads * qk_embed_size, 
			bias = False,
			device = device
		)
		self.key_embed: nn.Linear = nn.Linear(
			in_features = key_size,
			out_features = num_heads * qk_embed_size,
			bias = False,
			device = device
		)
		self.value_embed: nn.Linear = nn.Linear(
			in_features = value_size,
			out_features = num_heads * v_embed_size,
			bias = False,
			device = device
		)
		self.scale_dk: float = 1 / np.sqrt(qk_embed_size)
		self.softmaxing: nn.Softmax = nn.Softmax(dim = -1)
		self.num_heads: int = num_heads
		self.qk_size: int = qk_embed_size
		self.v_size: int = v_embed_size
		self.final_project: nn.Linear = nn.Linear(
			in_features = num_heads * v_embed_size,
			out_features = output_size,
			bias = False,
			device = device
		)
		
	def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
		# lemme reason on the final sizes assuming single input
		# query = h x n x dQK
		q_embed: torch.Tensor = torch.transpose(torch.reshape(self.query_embed(query), (-1, self.num_heads, self.qk_size)), 0, 1)
		# key = h x n x dQK
		k_embed: torch.Tensor = torch.transpose(torch.reshape(self.key_embed(key), (-1, self.num_heads, self.qk_size)), 0, 1)
		# value = h x n x dV
		v_embed: torch.Tensor = torch.transpose(torch.reshape(self.value_embed(value), (-1, self.num_heads, self.v_size)), 0, 1)
		# dot_product = h x n x n
		dot_product: torch.Tensor = self.scale_dk * torch.matmul(q_embed, torch.transpose(k_embed, -2, -1))
		dot_product -= torch.triu(torch.ones(dot_product.shape) * torch.inf, diagonal = 1)
		# softmaxed = h x n x n
		softmaxed: torch.Tensor = self.softmaxing(dot_product)
		# head_embed = h x n x dV
		head_embedded = torch.matmul(softmaxed, v_embed)
		# final output = n x output_size
		return self.final_project(torch.reshape(torch.transpose(head_embedded, 0, 1), (-1, self.v_size * self.num_heads)))	
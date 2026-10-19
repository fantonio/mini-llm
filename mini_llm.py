import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Hiperparâmetros
batch_size = 8         
block_size = 16        
max_iters = 3000       
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64            
n_head = 4             
n_layer = 3            
NOME_ARQUIVO_MODELO = "mini_gpt_treinado.pth"

# 2. Dataset Fixo
texto = "ser ou nao ser eis a questao para aprender a criar um modelo de linguagem do zero"
caracteres = sorted(list(set(texto)))
vocab_size = len(caracteres)

stoi = { ch:i for i,ch in enumerate(caracteres) }
itos = { i:ch for i,ch in enumerate(caracteres) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

dados = torch.tensor(encode(texto), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(dados) - block_size, (batch_size,))
    x = torch.stack([dados[i:i+block_size] for i in ix])
    y = torch.stack([dados[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 3. Componentes do Transformer (Arquitetura)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))    
        x = x + self.ffwd(self.ln2(x))  
        return x

class ModeloLinguagemGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device)) 
        x = self.blocks(x)    
        x = self.ln_f(x)      
        logits = self.lm_head(x) 
        loss = None if targets is None else F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1) 
        return idx

# =====================================================================
# 4. FLUXO AUTOMÁTICO DE DECISÃO (CHECA ARQUIVO OU TREINA)
# =====================================================================
model = ModeloLinguagemGPT().to(device)

if os.path.exists(NOME_ARQUIVO_MODELO):
    # Se o arquivo já existe no mesmo diretório
    print(f"🎉 Arquivo '{NOME_ARQUIVO_MODELO}' encontrado! Carregando pesos...")
    model.load_state_dict(torch.load(NOME_ARQUIVO_MODELO, map_location=device, weights_only=True))
    model.eval()
    print("Modelo carregado com sucesso.")
else:
    # Se o arquivo NÃO existe, inicia o treinamento completo do zero
    print(f"⚠️ Arquivo '{NOME_ARQUIVO_MODELO}' não encontrado.")
    print("Iniciando treinamento do zero...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for iter in range(max_iters):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iter % 500 == 0:
            print(f"  Iteração {iter}: Perda (Loss) = {loss.item():.4f}")
            
    # Salva para as próximas execuções
    torch.save(model.state_dict(), NOME_ARQUIVO_MODELO)
    print(f"💾 Treinamento concluído. Modelo salvo em '{NOME_ARQUIVO_MODELO}'.")
    model.eval()

# =====================================================================
# 5. GERAÇÃO DO TEXTO
# =====================================================================
prompt_usuario = "ser ou "
contexto_inicial = torch.tensor([encode(prompt_usuario)], dtype=torch.long, device=device)

print("\n--- TEXTO GERADO PELO MODELO ---")
# O [0] abaixo pega a primeira lista de números gerada e resolve o problema
lista_de_indices = model.generate(contexto_inicial, max_new_tokens=75).tolist()[0]
print(decode(lista_de_indices))
print("--------------------------------")

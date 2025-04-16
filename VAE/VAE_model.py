import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class SmEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, latent_dim: int, num_rnn_layers: int = 1): 
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # n_char: 46, n_feature: 128
        self.sequence_encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_rnn_layers, batch_first=True) 

        # A mapping function that projects the last hidden vector created by the LSTM into latent space
        self.mu_layer = nn.Linear(hidden_dim, latent_dim) # Q. 한 sample → latent_dim 차원 공간의 각 축마다 개별 정규분포를 예측 // 그렇게 해서 input의 분포를 근사?
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, token_indices:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_input = self.embedding_layer(token_indices) # (batch_size, seq_len, latent_dim)
        encoded_sequence, _ = self.sequence_encoder(embedded_input) # (batch_size, seq_len, latent_dim) 

        # Extract final latent vector per sequence (before padding token)
        latent_vectors = []
        for batch_idx in range(len(token_indices)):
            # batch_idx: i-th sample
            # token_indices[batch_idx]: length of i-th sample
            pad_index = 0 # To do: Managed by self.pad_index inside the SMILESDataset class
            valid_token_count = token_indices[batch_idx].ne(pad_index).sum().item() # Number of tokens excluding pad # item(): tensor -> int
            last_token_position = valid_token_count - 1
            latent_vectors.append(encoded_sequence[batch_idx, last_token_position, :]) # extract last (valid) hidden states
        latent_vectors = torch.stack(latent_vectors, dim=0) # [batch_size, latent_dim] # 만약 AE라면 그냥 이 latent vector를 return? # VAE니까 reparameterization 해줘서 정규 분포로 나타내는 것?
        
        mean = self.mu_layer(latent_vectors) # [batch_size, latent_dim]
        log_variance = self.logvar_layer(latent_vectors) # [batch_size, latent_dim]
        return mean, log_variance
    
    def reparameterize_trick(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        std_dev = torch.exp(0.5 * log_variance)
        random_noise = torch.randn_like(std_dev)
        sampled_latent = mean + random_noise * std_dev
        return sampled_latent
    

class SmDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, latent_dim: int, num_rnn_layers: int = 1):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # for target tokens to embedding
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm_decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.num_rnn_layers = num_rnn_layers

    def forward(self, latent_vector: torch.Tensor, target_tokens: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor: # target_tokens도 tensor?
        """Generate logit given latent vector `z` (for training)"""
        batch_size, seq_len = target_tokens.size() # target_tokens: [batch_size, seq_len]
        
        # LSTM should receive two hidden states (h_0, c_0)
        h_0 = self.latent_to_hidden(latent_vector).unsqueeze(0).repeat(self.num_rnn_layers, 1, 1) # [1, batch_size, hidden_dim] -> [num_rnn_layers, batch_size, hidden_dim] # repeat: repeat the same hidden state for each layer
        c_0 = torch.zeros_like(h_0)  # same shape
        hidden_state = (h_0, c_0) # LSTM should receive two hidden states (h_0, c_0)
        
        input_tokens = target_tokens[:, 0].unsqueeze(1) # [batch_size, 1] # <sos> token
        output_logits = []

        for t in range(1, seq_len):
            input_embedding = self.embedding_layer(input_tokens) # [batch_size, input_token_length, embedding_dim] # input_token_length = 1
            lstm_output, hidden_state = self.lstm_decoder(input_embedding, hidden_state) # lstm_output: [batch_size, 1, hidden_dim] 
            logits = self.output_projection(lstm_output.squeeze(1))  # [batch_size, vocab_size]
            
            output_logits.append(logits) 

            # Decide next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            next_input = target_tokens[:, t] if teacher_force else logits.argmax(dim=1) # To do: Modify logit.argmax -> softmax sampling
            input_tokens = next_input.unsqueeze(1) # [batch_size, 1]

        return torch.stack(output_logits, dim=1) # [batch_size, seq_len-1, vocab_size]
    
    def generate(self, latent_vector: torch.Tensor, max_length: int, start_token_idx: int = 1, end_token_idx: int = 2) -> torch.Tensor:
        """Generate sequence given latent vector `z` (for inference)"""
        batch_size = latent_vector.size(0) # latent_vactor: [batch_size, latent_dim]
        
        # LSTM should receive two hidden states (h_0, c_0)
        h_0 = self.latent_to_hidden(latent_vector).unsqueeze(0).repeat(self.num_rnn_layers, 1, 1) # [1, batch_size, hidden_dim]
        c_0 = torch.zeros_like(h_0)  # same shape
        hidden_state = (h_0, c_0)
        
        input_token = torch.full((batch_size, 1), start_token_idx, dtype=torch.long).to(latent_vector.device) # [batch_size, 1] # <sos> token # torch.long: 64-bit integer
        generated_sequence = []

        for _ in range(max_length):
            input_embedding = self.embedding_layer(input_token) # [batch_size, input_token_length, embedding_dim] # input_token_length = 1
            lstm_output, hidden_state = self.lstm_decoder(input_embedding, hidden_state) # lstm_output: [batch_size, 1, hidden_dim]
            logits = self.output_projection(lstm_output.squeeze(1))  # [batch_size, vocab_size]
            
            # Softmax sampling
            probs = nn.functional.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1] # sample from the distribution

            generated_sequence.append(next_token)            
            input_token = next_token

            if (next_token == end_token_idx).all(): # all samples generate <eos> token -> stop generation # To do: where is <pad>?
                break

        return torch.stack(generated_sequence, dim=1)  # [batch_size, generated_length]


class SmVAE(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, latent_dim: int, num_rnn_layers: int = 1):
        super().__init__()
        self.encoder = SmEncoder(vocab_size, embedding_dim, hidden_dim, latent_dim, num_rnn_layers)
        self.decoder = SmDecoder(vocab_size, embedding_dim, hidden_dim, latent_dim, num_rnn_layers)

        # special tokens
        self.pad_index = 0
        self.start_token_idx = 1
        self.end_token_idx = 0 # To do: replace with <eos> token index when model can generate <pad> token after <eos> token

        self.loss_function = CrossEntropyLoss(reduction='sum', ignore_index=self.pad_index)
        self.latent_dim = latent_dim
        self.num_rnn_layers = num_rnn_layers
        self.embedding_layer = self.encoder.embedding_layer # embedding layer is shared between encoder and decoder
    
    def forward(self, input_token_indices: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> tuple:
        mean, log_variance = self.encoder(input_token_indices)
        latent_vector = self.encoder.reparameterize_trick(mean, log_variance) # self: 현재 객체에 정의된 메서드를 호출하겠다는 의미

        target_tokens = input_token_indices[:, 1:]
        output_logits = self.decoder(latent_vector, target_tokens, teacher_forcing_ratio) 

        return output_logits, mean, log_variance

    def inference(self, latent_vector: torch.Tensor, max_length: int, start_token_idx: int = 1, end_token_idx: int = 2) -> torch.Tensor:
        return self.decoder.generate(latent_vector, max_length, start_token_idx, end_token_idx)

    def compute_loss(self, decoder_logits: torch.Tensor, input_tokens: torch.Tensor, mean: torch.Tensor, log_variance: torch.Tensor, length_batch:torch.Tensor) -> tuple[float, float]:
        total_reconstruction_loss = 0.0
        target_tokens = input_tokens[:, 1:] # [batch_size, seq_len-1] 

        # To do: modify not to use for loop
        for i in range(len(decoder_logits)): # i-th sample
            valid_token_count = length_batch[i] - 1 # -1: <sos> token # there is no <sos> token in target_tokens (+ decoder_logits)
            max_len = min(decoder_logits[i].size(0), target_tokens[i].size(0), valid_token_count) # If the sequence generated by the decoder is smaller than valid_token_count, a size mismatch error can occur.
            
            if max_len <= 0:
                continue
            # To do: Loss function becomes larger when frameshifted. consider rodge or edit distance
            total_reconstruction_loss += (self.loss_function(decoder_logits[i, :max_len, :], target_tokens[i, :max_len]) / max_len) # normalized by sequence length 
        total_reconstruction_loss /= len(decoder_logits) # len(decoder_logits) = batch_size

        # Compute KL divergence
        total_kl_divergence_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=1).mean() # [batch_size, latent_dim] # sum by latent_dim -> mean by batch_size
        return total_reconstruction_loss, total_kl_divergence_loss


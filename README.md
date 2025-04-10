# Tutorial_generative_model

모든 코드는 ```python3 scripts.py -h``` 하면 어떤 명령어를 입력해야 할지 알 수 있다. 

# 1. RNN model
- dataset: smiles.txt
- tokenizer: RNN_tokenizer.pkl 사용
- input: smiles sequence
- output: Mollogp 예측 (Regression)
- Mollogp?
![image](https://github.com/user-attachments/assets/902a0295-9dc9-43cd-8ad0-61e1a351d2d5)

# 2. VAE model
- dataset: smiles.txt
- tokenizer: tokenizer_w_special_tokens.pkl 사용
![image](https://github.com/user-attachments/assets/4bc965b2-7c0d-4dbe-b408-ec4bee6475bc)
## train
- input: smiles sequence
- output: smiles sequence

## test
- input: sampled latent vector
- output: generated smiles seqeunce

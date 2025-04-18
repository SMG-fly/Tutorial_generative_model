# Tutorial_generative_model

모든 코드는 ```python3 scripts.py -h``` 하면 어떤 명령어를 입력해야 할지 알 수 있다. 

# 1. RNN model
- dataset: smiles.txt
- tokenizer: RNN_tokenizer.pkl 사용
- input: smiles sequence
- output: Mollogp 예측 (Regression)
- Mollogp? 분자의 지용성 (lipophilicity) 을 나타내는 지표. 값이 높을수록 지용성, 낮을수록 수용성.
![image](https://github.com/user-attachments/assets/902a0295-9dc9-43cd-8ad0-61e1a351d2d5)
## Results
### wandb(train)
![image](https://github.com/user-attachments/assets/2fa2d904-1279-4ca8-b676-6908fe216988)



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

## Results
### wandb(train)
- VAE-training: 5epoch 돌린 결과
- VAE-training-2: 100epoch 돌린 결과
  ![image](https://github.com/user-attachments/assets/22b63a57-daca-4fa3-a099-c7f56941715d)


### Inference 결과
- 5epoch

![image](https://github.com/user-attachments/assets/c5c84f39-a164-4021-9b70-5cdde44ea703)

- 100epoch

![image](https://github.com/user-attachments/assets/e32ec862-a817-48fa-8429-e527ff9bcd30)


유의미한 결과를 도출하기 위해 더 큰 규모의 학습을 해야 할 것으로 보인다.



![image](https://github.com/user-attachments/assets/eed0ef47-e3d4-4076-b716-16c206339e3e)# Tutorial_generative_model

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

- softmax sampling 방식으로 1000epoch 돌린 결과

  ![image](https://github.com/user-attachments/assets/81e0a86f-8b84-4856-9fc1-d14cc945e212)



### Inference 결과
- 5epoch + argmax token

![image](https://github.com/user-attachments/assets/c5c84f39-a164-4021-9b70-5cdde44ea703)

- 100epoch + argmax token

![image](https://github.com/user-attachments/assets/e32ec862-a817-48fa-8429-e527ff9bcd30)

- 1000epoch + softmax token

![image](https://github.com/user-attachments/assets/9b87ea4b-2c9e-4175-908c-69457360d0c7)

 결과의 다양성을 높이기 위해 logit값이 max인 token을 선택하는 것이 아니라 softmax를 통과시켜 확률적으로 token이 sampling 되도록했다. 

softmax sampling으로 변경한 결과 다양한 smiles 식이 나오기 시작했지만, 여전히 valid한 분자는 하나도 나오지 않았다. 

# 3. MPNN
dataset: smiles.txt
output: molecular properties (logP, QED)

## Preprocessing
input: smiles.txt
output: graph.pt
script: MPNN_preprocess.py

## train & test
input: graph.pt
output: molecular properties (logP, QED)

## Result
### wandb
- logP: 노란색(10epoch) -> 청록색 (이후 200epoch)
- QED: 주황색
- 200epoch
- loss를 MSE(reduction="sum")으로 해두어서 해당 dimension만큼 큰 loss가 wandb에 찍혔다. 현재 MPNN_train 코드는 mean loss로 수정했으나, 이전 코드(RNN, VAE)도 수정이 필요하다.
- 대부분의 loss 변화는 학습 초반에 있었다. 

![image](https://github.com/user-attachments/assets/dc5743c8-9162-464d-b189-95e49c1a56cd)


### Inference
- logP

![image](https://github.com/user-attachments/assets/3f02b882-0687-4c0e-9300-6ebc2b78cb18)

- QED

![image](https://github.com/user-attachments/assets/4d2b1a5e-fbfe-46d5-b9c4-4998fc9c6fa8)



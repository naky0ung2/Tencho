# 데이터 살펴보기
import pandas as pd # 데이터프레임 다룰때 
import os # operation system의 약자로 운영체제의 기능을 파이썬에서도 사용할 수 있도록 함
import string # 알파벳, 숫자, 특수문자 그리고 공백과 같은 문자 집합을 제공

df = pd.read_csv("./ArticlesApril2017.csv") # csv 파일 불러오기
print(df.columns) # 컬럼명 출력

# 학습용 데이터셋 정의
import numpy as np # 수식, 배열 계산에 쓰는 라이브러리 
import glob # 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환 

from torch.utils.data.dataset import Dataset # 전체 데이터셋을 만듦 

class TextGeneration(Dataset): # 클래스 정의
    def clean_text(self, txt): # 함수 정의
        # 모든 단어를 소문자lower()로 바꾸고 특수문자 punctuation(온점, 따옴표 )를 제거
        txt = "".join(v for v in txt if v not in string.punctuation).lower()
        return txt # 값 리턴
    def __init__(self):
        all_headlines = [] # 빈 리스트 미리 만들어두고 

        for filename in glob.glob("./*.csv"): # 모든 헤드라인의 텍스트를 불러옴
            if 'Articles' in filename:        # filename에 'Articles' 이 있으면
                article_df = pd.read_csv(filename) # 파일 읽기 
                all_headlines.extend(list(article_df.headline.values)) # 데이터셋의 headline의 값을 all_headlines에 추가
                break

        all_headlines = [h for h in all_headlines if h != "Unknown"] # headline 중 unknown 값은 제거
       
        self.corpus = [self.clean_text(x) for x in all_headlines]    # 구두점 제거 및 전처리가 된 문장들을 리스트로 반환
        self.BOW = {}  # 

        # 모든 문장의 단어를 추출해 고유번호 지정
        for line in self.corpus: # 리스트에 들어간 애들 중에 한 라인씩
            for word in line.split(): #  .split() 으로 쪼개기
                if word not in self.BOW.keys(): # word가 BOW 키 값에 없으면 
                    self.BOW[word] = len(self.BOW.keys()) # self.BOW.keys()의 크기가 BOW[word]..?

        # 모델의 입력으로 사용할 데이터
        self.data = self.generate_sequence(self.corpus) # 데이터 설정
    def generate_sequence(self, txt):
        seq = []

        for line in txt: # txt를 for문 돌림
            line = line.split() # line을 쪼갬 , 문자열을 일정한 규칙으로 잘라서 리스트로 만들어 줌
            line_bow = [self.BOW[word] for word in line] # line 에서 다시 for문 

            data = [([line_bow[i], line_bow[i+1]], line_bow[i+2])  # 단어 2개를 입력으로, 그다음 단어를 정답으로
            for i in range(len(line_bow)-2)] # line_bow-2 만큼 반복 
            
            seq.extend(data) # seq에 넣기

        return seq # 리스트로 출략
    
    def __len__(self): # 데이터 길이
        return len(self.data)
    
    def __getitem__(self, i): # 클래스 인덱스에 접근
        data = np.array(self.data[i][0])  # 입력 데이터
        label = np.array(self.data[i][1]).astype(np.float32)  # 출력 데이터

        return data, label
    
# LSTM모델 정의
import torch # 토치 라이브러리
import torch.nn as nn # 토치 기본 모듈


class LSTM(nn.Module): # LSTM 클래스 정의
   def __init__(self, num_embeddings):
       super(LSTM, self).__init__() # nn.Module 클래스의 요소 상속
      
       self.embed = nn.Embedding(  # 밀집표현을 위한 임베딩층
           num_embeddings=num_embeddings, embedding_dim=16) # mbedding_dim : 밀집 표현의 차원을 의미
       
       # LSTM을 5개층을 쌓음
       self.lstm = nn.LSTM( 
           input_size=16,  # 인풋 사이즈
           hidden_size=64, # 은닉층의 사이즈에 해당
           num_layers=5,  # 레이어층 5개
           batch_first=True) # batch_size가 제일 먼저 앞으로 이동 / 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
       
       self.fc1 = nn.Linear(128, num_embeddings)  # 분류를 위한 MLP층
       self.fc2 = nn.Linear(num_embeddings,num_embeddings)  # 분류를 위한 MLP층

       self.relu = nn.ReLU()   # 활성화 함수, 렐루

   def forward(self, x): # Network의 forward를 정의
       x = self.embed(x) # 위에서 만든 임베딩층

       # LSTM 모델의 예측값
       x, _ = self.lstm(x)
       x = torch.reshape(x, (x.shape[0], -1)) #  텐서의 모양을 변경, 원본 tensor의 복사본을 받음
       x = self.fc1(x)  # 분류를 위한 MLP층
       x = self.relu(x) # 활성화 함수
       x = self.fc2(x)  # 분류를 위한 MLP층

       return x # 값 반환
   
# 모델 학습하기
import tqdm # 진행률 보여줌
from torch.utils.data.dataloader import DataLoader # 데이터셋을 순회할 수 있도록 함
from torch.optim.adam import Adam # 최적화

# 학습을 진행할 프로세서 정의
device = "cuda" if torch.cuda.is_available() else "cpu" # 쿠다 가능하면 gpu 써라 아님 cpu

dataset = TextGeneration()  # 데이터셋 정의
model = LSTM(num_embeddings=len(dataset.BOW)).to(device)  # 모델 정의
loader = DataLoader(dataset, batch_size=64) # 데이터 불러옴
optim = Adam(model.parameters(), lr=0.001)  # 옵티마이저, 학습률 지정

for epoch in range(200): # 200번 학습시킬 거
   iterator = tqdm.tqdm(loader) # tqdm으로 진행사황 나타냄, 반복 
   for data, label in iterator: # 데이터, 라벨 for문 돌리기
      
       optim.zero_grad()  # 기울기 초기화
       pred = model(torch.tensor(data, dtype=torch.long).to(device))  # 모델의 예측값 
       loss = nn.CrossEntropyLoss()( # torch.long :  정수를 사용
           pred, torch.tensor(label, dtype=torch.long).to(device))    # 정답 레이블은 long 텐서로 반환해야 함
       
       loss.backward()  # 오차 역전파
       optim.step()

       iterator.set_description(f"epoch{epoch} loss:{loss.item()}") # 에폭, 로스 출력하도록 함

torch.save(model.state_dict(), "lstm.pth") # 모델 저장 

def generate(model, BOW, string="finding an ", strlen=10):
    device = "cuda" if torch.cuda.is_available() else "cpu" # gpu 쓸 수 있음 쓰겠다.

    print(f"input word: {string}") # 인풋 월드 : string 이다를 출력

    with torch.no_grad(): # 그레디언트 비활성화
       for p in range(strlen):
           
           words = torch.tensor( # 입력 문장을 텐서로 변경
               [BOW[w] for w in string.split()], dtype=torch.long).to(device)

           
           input_tensor = torch.unsqueeze(words[-2:], dim=0) # 차원을 추가 
           output = model(input_tensor)  # 모델을 이용해 예측
           output_word = (torch.argmax(output).cpu().numpy()) # input tensor에 있는 모든 element들 중에서 가장 큰 값을 가지는 공간의 인덱스 번호를 반환하는 함수
           string += list(BOW.keys())[output_word]  # 문장에 예측된 단어를 추가
           string += " " # 띄어쓰기 ? 

    print(f"predicted sentence: {string}") # 예측한 문장 출력이 string ~ 값이라고 출력

model.load_state_dict(torch.load("lstm.pth", map_location=device)) # 저장된 모델 불러오기 
pred = generate(model, dataset.BOW) # 예측하기 
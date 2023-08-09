# 데이터 확인해보기
import glob # 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
import matplotlib.pyplot as plt # 그래프 그리는 라이브러리
from PIL import Image # 이미지 파일 불러올 떄 사용 

# 데이터셋에 포함된 파일명을 불러옴
imgs = glob.glob("/Users/nakyoung/Desktop/test/*.jpg") # 그냥 *라고 주면 모든 파일과 디렉터리를 볼 수 있음

for i in range(4): # 4번 반복
    img = Image.open(imgs[i]) #이미지를 불러오자
    plt.subplot(2,2,i+1) # 2 x 2 로  
    plt.imshow(img) # 그래프 그리기
plt.show() # 그래프 창 띄워줘

# RGB를 LAB로 변환하는 함수
import cv2 # opencv 라이브러리 
import numpy as np #array, 수학적 계산에 사용 
from torch.utils.data.dataset import Dataset # dataset과 dataloader 기능을 기반으로 미니배치 학습, 데이터 셔플, 병렬 처리 구현 가능, 전체 데이셋을 만드는 과정 

# RGB를 LAB로 변환
def rgb2lab(rgb): # 함수를 만들자
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB) # opencv를 이용해서 rgb를 lab로 바꾸자

# LAB를 RGB로 변환
def lab2rgb(lab): # 함수를 만들자 
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) # LAB를 RGB로 바꾸자 

# 학습용 데이터셋 정의
class AutoColoring(Dataset) : # 학습에 이용할 데이터셋 객체
    def __init__(self) :      # __init__(self) : 필요한 변수들을 선언하는 메서드. input으로 오는 x와 y를 load 하거나, 파일목록을 load한다.
        self.data = glob.glob("/Users/nakyoung/Desktop/test/*.jpg") # 데이터 지정
        
    def __len__(self) : # 데이터 크기 출력하는 함수
        return len(self.data) # 데이터 크기 리턴
    
    def __getitem__(self, i) : # 슬라이싱을 구현하기 위해서 필요
        rgb = np.array(Image.open(self.data[i].resize(256,256))) # 이미지 불러와서 리사이즈하고 넘파이 어레이로 저장
        lab = rgb2lab(rgb) # rgb를 lab로 바꿈
        lab = lab.transpose((2, 0 ,1).astype(np.float32)) # 보통 [width, height, channels] -> [channels , width, height]
        
        return lab[0] # 채널수, lab[1:] # width , height 반환
    
import torch # 토치 라이브러리 
import torch.nn as nn # 토치 기본 모듈


class LowLevel(nn.Module): # 로우 레벨 특징 추출기 정의
   def __init__(self):
       # 로우 레벨 특징 추출기를 구성하는 층의 정의
       super(LowLevel, self).__init__()

       self.low1 = nn.Conv2d(1, 64,  # 합성곱 , 인풋 채널 1 , 아웃풋 64, 커널 사이즈, 스트라이드 , 패딩 지정
                             kernel_size=3, stride=2, padding=1)
       self.lb1 = nn.BatchNorm2d(64)   # 배치 정규화 정의
       self.low2 = nn.Conv2d(64, 128,  # 합성곱 , 인풋 채널 64 , 아웃풋 128, 커널 사이즈, 스트라이드 , 패딩 지정
                             kernel_size=3, stride=1, padding=1)
       self.lb2 = nn.BatchNorm2d(128) # 배치 정규화 정의
       self.low3 = nn.Conv2d(128, 128, # 합성곱 , 인풋 채널 128 , 아웃풋 128, 커널 사이즈, 스트라이드 , 패딩 지정
                             kernel_size=3, stride=2, padding=1)
       self.lb3 = nn.BatchNorm2d(128) # 배치 정규화 정의
       self.low4 = nn.Conv2d(128, 256,  # 합성곱 , 인풋 채널 128 , 아웃풋 256, 커널 사이즈, 스트라이드 , 패딩 지정
                             kernel_size=3, stride=1, padding=1)
       self.lb4 = nn.BatchNorm2d(256) # 배치 정규화 정의
       self.low5 = nn.Conv2d(256, 256,   # 합성곱 , 인풋 채널 256 , 아웃풋 256, 커널 사이즈, 스트라이드 , 패딩 지정
                             kernel_size=3, stride=2, padding=1)
       self.lb5 = nn.BatchNorm2d(256) # 배치 정규화 정의
       self.low6 = nn.Conv2d(256, 512, # 합성곱 , 인풋 채널 256 , 아웃풋 512, 커널 사이즈, 스트라이드 , 패딩 지정
                             kernel_size=3, stride=1, padding=1)
       self.lb6 = nn.BatchNorm2d(512) # 배치 정규화 정의

       self.sigmoid = nn.Sigmoid() # 활성화 함수는 시그모이드
       
   def forward(self, x): # forward 함수 정의
       # 기본 블록 구성
       low = self.low1(x)        #   합성곱
       low = self.lb1(low)       #   배치 정규화
       low = self.sigmoid(low)   #   시그모이드
   
       # 위의 구조 반복, 6개 까지 
       low = self.low2(low)      
       low = self.lb2(low)       
       low = self.sigmoid(low)   

       low = self.low3(low)
       low = self.lb3(low)
       low = self.sigmoid(low)

       low = self.low4(low)
       low = self.lb4(low)
       low = self.sigmoid(low)

       low = self.low5(low)
       low = self.lb5(low)
       low = self.sigmoid(low)
       
       low = self.low6(low)
       low = self.lb6(low)
       low = self.sigmoid(low)

       return low
   
# 미들 레벨 특징 추출기 정의
class MidLevel(nn.Module):
    def __init__(self):
       # 미들 레벨 특징 추출기를 구성하는 층의 정의
       super(MidLevel, self).__init__()

       self.mid1 = nn.Conv2d(512, 512,  # 합성곱 , 인풋 채널 512 , 아웃풋 512, 커널 사이즈, 스트라이드 , 패딩 지정
                             kernel_size=3, stride=1, padding=1)
       self.mb1 = nn.BatchNorm2d(512)   # 배치 정규화 
       self.mid2 = nn.Conv2d(512, 256,  # 합성곱 , 인풋 채널 512 , 아웃풋 256, 커널 사이즈, 스트라이드 , 패딩 지정
                             kernel_size=3, stride=1, padding=1)
       self.mb2 = nn.BatchNorm2d(256)   # 배치 정규화

       self.sigmoid = nn.Sigmoid()      # 활성화 함수 시그모이드 사용

    def forward(self, x):
       # 미들 레벨 특징 추출기의 기본 블록
       mid = self.mid1(x)       #  합성곱
       mid = self.mb1(mid)      #  배치 정규화
       mid = self.sigmoid(mid)  #  시그모이드

       mid = self.mid2(mid)     #  합성곱
       mid = self.mb2(mid)      #  배치 정규화
       mid = self.sigmoid(mid)  #  시그모이드

       return mid
   
# 글로벌 레벨 특징 추출기 정의
class GlobalLevel(nn.Module):
    def __init__(self):
       super(GlobalLevel, self).__init__()

       self.glob1 = nn.Conv2d(512, 512,  # 합성곱 , 인풋 채널 512 , 아웃풋 512, 커널 사이즈, 스트라이드 , 패딩 지정
                              kernel_size=3, stride=2, padding=1)
       self.gb1 = nn.BatchNorm2d(512) # 배치 정규화 
       self.glob2 = nn.Conv2d(512, 512,  # 합성곱 , 인풋 채널 512 , 아웃풋 512, 커널 사이즈, 스트라이드 , 패딩 지정
                              kernel_size=3, stride=1, padding=1)
       self.gb2 = nn.BatchNorm2d(512)     # 배치 정규화 
       self.glob3 = nn.Conv2d(512, 512,  # 합성곱 , 인풋 채널 512 , 아웃풋 512, 커널 사이즈, 스트라이드 , 패딩 지정
                              kernel_size=3, stride=2, padding=1)
       self.gb3 = nn.BatchNorm2d(512)    # 배치 정규화 
       self.glob4 = nn.Conv2d(512, 512,   # 합성곱 , 인풋 채널 512 , 아웃풋 512, 커널 사이즈, 스트라이드 , 패딩 지정
                              kernel_size=3, stride=1, padding=1)
       self.gb4 = nn.BatchNorm2d(512)     # 배치 정구화
       
       # 여기서는 분류기로 사용되는것이 아닌, 색을 칠하기 위해 사용하는 특징으로 사용
       self.fc1 = nn.Linear(in_features=32768, out_features=1024) # 글로벌 레벨 특징 추출기의 MLP층 구성
       self.fc2 = nn.Linear(in_features=1024, out_features=512)
       self.fc3 = nn.Linear(in_features=512, out_features=256)

       self.sigmoid = nn.Sigmoid() # 활성화함수
       
    def forward(self, x):
       # 글로벌 레벨 특징 추출기의 기본 블록
       glo = self.glob1(x)         # 합성곱
       glo = self.gb1(glo)         # 배치 정규화
       glo = self.sigmoid(glo)     # 활성화

       # 위의 구조 똑같이 반복
       glo = self.glob2(glo)
       glo = self.gb2(glo)
       glo = self.sigmoid(glo)

       glo = self.glob3(glo)
       glo = self.gb3(glo)
       glo = self.sigmoid(glo)

       glo = self.glob4(glo)
       glo = self.gb4(glo)
       glo = self.sigmoid(glo)
      
       # flatten : 추출된 특징을 1차원으로 펼쳐줌
       glo = torch.flatten(glo, start_dim=1) # 1차원 부터
       glo = self.fc1(glo) # 첫번째 linear 층
       glo = self.sigmoid(glo) # 활성화 함수 
       glo = self.fc2(glo) #  두번째 linear 층
       glo = self.sigmoid(glo) # 활성화 함수 
       glo = self.fc3(glo) # 세번째 linear 층
       glo = self.sigmoid(glo) # 활성화 함수

       return glo
   
# 컬러라이제이션 네크워크 정의
class Colorization(nn.Module):
    def __init__(self):
       super(Colorization, self).__init__()
       # Colorization 네트워크 구성에 필요한 층의 정의
       # 업샘플링 커널:3 스트라이드:1 패딩:1
       self.color1 = nn.ConvTranspose2d(256, 128, 3, 1, 1) # 줄어들었던 사이즈 늘리는 과정
       self.cb1 = nn.BatchNorm2d(128)
       
       # 업샘플링 커널:2 스트라이드:2 패딩:0
       self.color2 = nn.ConvTranspose2d(128, 64, 2, 2)
       self.cb2 = nn.BatchNorm2d(64)
       
       # 업샘플링 커널:3 스트라이드:1 패딩:1
       self.color3 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
       self.cb3 = nn.BatchNorm2d(64)
       # 업샘플링 커널:2 스트라이드:2 패딩:0
       self.color4 = nn.ConvTranspose2d(64, 32, 2, 2)
       self.cb4 = nn.BatchNorm2d(32)
       # 업샘플링 커널:2 스트라이드:2 패딩:0
       self.color5 = nn.ConvTranspose2d(32, 2, 2, 2)

       self.sigmoid = nn.Sigmoid()
       
    def forward(self, x): #  forward 정의
       color = self.color1(x) # color1 층
       color = self.cb1(color) # 정규화
       color = self.sigmoid(color) # 활성화함수 
       # 반복
       color = self.color2(color)
       color = self.cb2(color)
       color = self.sigmoid(color)
       color = self.color3(color)
       color = self.cb3(color)
       color = self.sigmoid(color)
       color = self.color4(color)
       color = self.cb4(color)
       color = self.sigmoid(color)
       color = self.color5(color)

       return color
   
# 자동채색 모델 정의
class AutoColoringModel(nn.Module):
    def __init__(self):
       super(AutoColoringModel, self).__init__()
       self.low = LowLevel()    # 로우 레벨 특징 추출기
       self.mid = MidLevel()    # 미들 레벨  특징 추출기
       self.glob = GlobalLevel() # 글로벌 레벨 특징 추출기

       self.fusion = nn.Conv2d(512, 256,  # 특징 합치기
                               kernel_size=3, stride=1, padding=1)
    
       self.color = Colorization() # 색 입히기

       
       self.sigmoid = nn.Sigmoid() # 활성화 함수
    def forward(self, x):  # 로 레벨 특징 추출기로 입력
       low = self.low(x)

       # 로 레벨 특징 추출기의 출력을 넣어줌
       mid = self.mid(low)
       glo = self.glob(low)
       
       fusion = glo.repeat(1, mid.shape[2]*mid.shape[2]) # 글로벌 레벨 특징 추출기의 출력을 미들 레벨 특징 추출기의 출력 크기가 되도록 반복
       fusion = torch.reshape( # 텐서 크기 바꾸기
           fusion, (-1, 256, mid.shape[2], mid.shape[2]))
       fusion = torch.cat([mid, fusion], dim=1) # cat : 글로벌 레벨 특징 추출기의 특징과 미들 레벨 특징 추출기의 특징을 결합
       fusion = self.fusion(fusion)
       fusion = self.sigmoid(fusion)

       color = self.color(fusion)  # 컬러라이제이션 네크워크

       return color
   
# 모델 학습하기
import tqdm # 진행바 표시

from torch.utils.data.dataloader import DataLoader # 데이터 다룰때 사용
from torch.optim.adam import Adam # 옵티마이저

device = "cuda" if torch.cuda.is_available() else "cpu" # 쿠다 가능하면 gpu 사용

model = AutoColoringModel().to(device) # 모델 정의



dataset = AutoColoring() # 데이터 정의
loader = DataLoader(dataset, batch_size=32, shuffle=True) # 배치 사이즈 지정, 셔플 ok 
optim = Adam(params=model.parameters(), lr=0.01) # 옵티마이저 지정


# 학습 루프 정의
for epoch in range(200): # 200번 학습
   iterator = tqdm.tqdm(loader)
   for L, AB in iterator:
       # L 채널은 흑백 이미지 이므로 채널 차원을 확보해야 함
       L = torch.unsqueeze(L, dim=1).to(device) # 차원 늘리기 
       optim.zero_grad() # 초기화
      
       # A, B 채널을 예측
       pred = model(L)
      
       # 손실 계산과 오차 역전파
       loss = nn.MSELoss()(pred, AB.to(device)) # mse 사용 
       loss.backward()
       optim.step()

       iterator.set_description(f"epoch:{epoch} loss:{loss.item()}")

# 모델 가중치 저장
torch.save(model.state_dict(), "AutoColor.pth")

# 모델 성능 평가하기


test_L, test_AB = dataset[0] # 결과 비교를 위한 실제 이미지
test_L = np.expand_dims(test_L, axis=0) # pyplot의 이미지 형식에 맞추기 위헤 뱐형
real_img = np.concatenate([test_L, test_AB])
real_img = real_img.transpose(1, 2, 0).astype(np.uint8)
real_img = lab2rgb(real_img)

# 모델이 예측한 결과
with torch.no_grad():
   # 모델 가중치 불러오기
   model.load_state_dict(
       torch.load("AutoColor.pth", map_location=device))

   # 모델의 예측값 계산
   input_tensor = torch.tensor(test_L)
   input_tensor = torch.unsqueeze(input_tensor, dim=0).to(device) # 차원을 늘림
   pred_AB = model(input_tensor)

   # pyplot의 이미지 형식에 맞추기 위한 약간의 변형이 필요함
   pred_LAB = torch.cat([input_tensor, pred_AB], dim=1)
   pred_LAB = torch.squeeze(pred_LAB) #  Tensor의 차원을 줄이는 함수
   pred_LAB = pred_LAB.permute(1, 2, 0).cpu().numpy() #  permute : 배열의 차원을 벡터 dimorder 에 지정된 순서대로 재배열
   pred_LAB = lab2rgb(pred_LAB.astype(np.uint8))

# 실제와 예측값의 비교
plt.subplot(1, 2, 1)
plt.imshow(real_img) # 실제 이미지
plt.title("real image")
plt.subplot(1, 2, 2)
plt.imshow(pred_LAB) # 예측 이미지
plt.title("predicted image")
plt.show()
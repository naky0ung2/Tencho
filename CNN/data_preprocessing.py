import torch # facebook에서 제공하는 딥러닝 도구, pytorch 사용하려고 불러오기
import matplotlib.pyplot as plt     # matplotlib :그래프 그리는 기본 라이브러리, pyplot : 모듈의 각각의 함수를 사용해서 그래프를 그릴 수 있음
import torchvision.transforms as T  # torchvision이 제공하는 이미지 변환의 기초 기능 

from torchvision.datasets.cifar import CIFAR10  # torchvision.datasets : torchvision이 가지고 있는 데이터셋, Train과 Test 데이터셋이 원래 나눠져 있음
from torchvision.transforms import Compose      # Compose() :하나의 함수로 구성 가능                                            
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomCrop, Normalize 
# [Totensor : 이미지를 토치의 텐서로 변환, RandomHorizontalFlip : y축 기준으로 대칭,  RandomCrop : 랜덤으로 이미지 일부 제거 , Normalize : 정규화 , Centercrop도 있음]
 

transforms = Compose([  # torchvision이 제공하는 이미지 변환의 기초 기능 , 데이터 전처리 함수, tf를 입력받아 차례대로 실행
   T.ToPILImage(),      #  Tensor에서 이미지로 변환
   RandomCrop((32, 32), padding=4), # 이미지 일부를 제거한 뒤 (32,32) 크기로 복원, padding=4 모자란 부분을 0으로 채우자
   RandomHorizontalFlip(p=0.5),     # p 확률로 이미지를 좌우대칭
   T.ToTensor(),  # 이미지를 Tensor로
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),  # 평균=mean,표준편차=std를 갖도록 정규화
   T.ToPILImage() # Tensor를 이미지로
])

training_data = CIFAR10(    # CIFAR-10 데이터셋 불러오기 
    root="./",              # 데이터 다운 받을 경로 지정
    train=True,             # True 이면 학습 데이터로 쓰겠다.
    download=True,          # 데이터를 다운 받을 것임 
    transform=transforms)   # 이미지를 파이토치 텐서로 변환

test_data = CIFAR10(        # CIFAR-10 데이터셋 불러오기
    root="./",              # 데이터 다운 받을 경로 지정
    train=False,            # False 이면 테스트용 데이터로 쓰겠다.
    download=True,          # 다운로드 받겠다.
    transform=transforms)   # 이미지를 파이토치 텐서로 변환

for i in range(9):          # 9번 반복
   plt.subplot(3, 3, i+1)   # 3x3 으로 plot 띄울것
   plt.imshow(training_data.data[i]) # training_data의 0~8 번째 순서대로 이미지 보여주자
   
plt.show() # 이미지 띄워서 보여주기

############################# 데이터셋의 평균과 표준 편차 ############################# 

training_data = CIFAR10(  # CIFAR-10 데이터셋 불러오기
    root="./",            # 데이터 다운 받을 경로 지정
    train=True,           # True 이면 학습 데이터로 쓰겠다.
    download=True,        # 데이터를 다운 받을 것임
    transform=ToTensor()) # 이미지를 파이토치 텐서로 변환


imgs = [item[0] for item in training_data] # item[0]은 이미지, item[1]은 정답 레이블 , 이미지를 여러개 담고 있는 리스트

# --> 리스트를 이미지를 여러개 담고 있는 텐서로 바꿔줘야함 --> stack 사용!

imgs = torch.stack(imgs, dim=0).numpy() # imgs를 하나로 합침 , tensor를 dim 방향으로 합침

# rgb 각각의 평균
mean_r = imgs[:,0,:,:].mean() 
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()
print(mean_r,mean_g,mean_b) # 평균 프린트

# rgb 각각의 표준편차
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()
print(std_r,std_g,std_b) # 표준편차 프린트
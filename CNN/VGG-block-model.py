import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torchvision.transforms import Compose  # Compose() :하나의 함수로 구성 가능 
from torchvision.datasets.cifar import CIFAR10  # torchvision.datasets : torchvision이 가지고 있는 데이터셋, Train과 Test 데이터셋이 원래 나눠져 있음
from torch.utils.data.dataloader import DataLoader          
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomCrop, Normalize 

class BasicBlock(nn.Module): # 기본 블록을 정의
   # 기본블록을 구성하는 계층의 정의
   def __init__(self, in_channels, out_channels, hidden_dim):
       
       super(BasicBlock, self).__init__() # nn.Module 클래스의 요소 상속

       
       self.conv1 = nn.Conv2d(in_channels, hidden_dim,  # 합성곱층 정의
                              kernel_size=3, padding=1)
       self.conv2 = nn.Conv2d(hidden_dim, out_channels, 
                              kernel_size=3, padding=1)
       self.relu = nn.ReLU()

       
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # stride는 커널의 이동 거리를 의
  
   def forward(self, x): # 기본블록의 순전파 정의
       x = self.conv1(x)
       x = self.relu(x)
       x = self.conv2(x)
       x = self.relu(x)
       x = self.pool(x)
      
       return x
   
class CNN(nn.Module):
    def __init__(self, num_classes): # num_classes는 클래스의 개수
       super(CNN, self).__init__()
       
       self.block1 = BasicBlock(in_channels=3, out_channels=32, hidden_dim=16) # 합성곱 기본 블록의 정의
       self.block2 = BasicBlock(in_channels=32, out_channels=128, hidden_dim=64)
       self.block3 = BasicBlock(in_channels=128, out_channels=256, 
                                hidden_dim=128)

       self.fc1 = nn.Linear(in_features=4096, out_features=2048)  # 분류기 정의
       self.fc2 = nn.Linear(in_features=2048, out_features=256)
       self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

       self.relu = nn.ReLU() # 분류기의 활성화 함수

    def forward(self, x):
       x = self.block1(x)
       x = self.block2(x)
       x = self.block3(x)  # 출력 모양: (-1, 256, 4, 4) 
       x = torch.flatten(x, start_dim=1) # 2차원 피처맵을 1차원으로

       x = self.fc1(x)
       x = self.relu(x)
       x = self.fc2(x)
       x = self.relu(x)
       x = self.fc3(x)

       return x

######################## 데이터 증강 정의 ###############################

transforms = Compose([
   RandomCrop((32, 32), padding=4),  # 랜덤 크롭핑
   RandomHorizontalFlip(p=0.5),      # p의 확률로 y축으로 뒤집기
   ToTensor(),                       # 텐서로 변환
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))  # 이미지 정규화
])

######################## 데이터 로드 및 모델 정의 ##########################

training_data = CIFAR10(root="./", train=True, download=True, transform=transforms) # 학습 데이터 불러오기 
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)    # 테스트 데이터 불러오기

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)  # 학습데이터 데이터로더 정의
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)      # 학습데이터 데이터로더 정의

device = "cuda" if torch.cuda.is_available() else "cpu" # 학습을 진행할 프로세서 설정

model = CNN(num_classes=10) # CNN 모델 정의
model.to(device) # 모델을 device로 보냄

######################## 모델 학습 ##########################

lr = 1e-3 # 학습률 정의
optim = Adam(model.parameters(), lr=lr) # 최적화 기법 정의

for epoch in range(100): # 학습 epoch 정의
   for data, label in train_loader:  # 데이터 호출
       optim.zero_grad()  # 기울기 초기화
       preds = model(data.to(device))  # 모델의 예측
       loss = nn.CrossEntropyLoss()(preds, label.to(device))  # 오차역전파와 최적화
       loss.backward() 
       optim.step() 

   if epoch==0 or epoch%10==9:  # 10번마다 손실 출력
       print(f"epoch{epoch+1} loss:{loss.item()}")


torch.save(model.state_dict(), "CIFAR.pth") # 모델 저장

######################## 모델 성능 평가하기 #############################

model.load_state_dict(torch.load("CIFAR.pth", map_location=device))

num_corr = 0

with torch.no_grad():
   for data, label in test_loader:

       output = model(data.to(device))
       preds = output.data.max(1)[1]
       corr = preds.eq(label.to(device).data).sum().item()
       num_corr += corr

   print(f"Accuracy:{num_corr/len(test_data)}")

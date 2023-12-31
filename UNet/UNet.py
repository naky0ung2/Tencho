# 입력과 정답 이미지 확인하기
import matplotlib.pyplot as plt # 그래프 그리는 라이브러리
from PIL import Image           # 이미지 분석 및 처리를 쉽게 할 수 있는 라이브러리

path_to_annotation = \
    "/Users/nakyoung/Downloads/unet_annotations/trimaps" # annotation 경로 
path_to_image = \
    "/Users/nakyoung/Downloads/unet_images" # 이미지 저장 경로 
    
annotation = Image.open(path_to_annotation + "/pomeranian_173.png") # 이미지 불러오기 

plt.subplot(1,2,1) # 그래프 그리기 
plt.title("annotation") # 제목 지정
plt.imshow(annotation)  # 그래프 띄우기 

image = Image.open(path_to_image + "/pomeranian_173.jpg") # 이미지 불러오기 
plt.subplot(1,2,2) # 그래프 그리기
plt.title("image") # 제목 지정
# plt.imshow(image)

# plt.show()

# 데이터 셋 만들기
import glob  # glob : 파일들의 리스트를 뽑을 때 사용 , 특정 경로에 존재하는 파일 or 디렉토리의 리스트를 불러옴
import torch # 토치 라이브러리 
import numpy as np # 배열 연산에 사용 
 
from torch.utils.data.dataset import Dataset # 데이터셋 사용할 때 쓰는 라이브러리 

class Pets(Dataset) : # 데이터셋 사용 위한 라이브러리 
    def __init__(self, path_to_img,
                 path_to_anno, 
                 train=True,
                 transforms = None,
                 input_size=(128,128)) :
        
        self.images = sorted(glob.glob(path_to_img + '/*.jpg')) # 정답과 입력 이미지를 이름순으로 정렬
        self.annotations = sorted(glob.glob(path_to_anno + '/*.png'))
        
        self.X_train = self.images[:int(0.8* len(self.images))] # 데이터셋을 학습과 평가용으로 나눔, train 80%
        self.X_test = self.images[int(0.8* len(self.images)):]  # test 20%
        self.Y_train = self.annotations[:int(0.8 * len(self.annotations))] # annotation data도 마찬가지로 train 나눔
        self.Y_test = self.annotations[int(0.8 * len(self.annotations)):]  # test도 
        
        self.train = train # 학습용 데이터, 평가용 데이터 결정 여부
        self.transforms = transforms # 사용할 데이터 늘리기
        self.input_size = input_size # 입력 이미지 크기
        
    def __len__(self): # 데이터 갯수
        if self.train :
            return len(self.X_train) # 학습데이터 길이
        else :
            return len(self.X_test)  # test 데이터 길이
        
    def preprocess_mask(self, mask):        # 정답을 변환해주는 함수
       mask = mask.resize(self.input_size)  # input 사이즈로 변경 
       mask = np.array(mask).astype(np.float32) # 배열 데이터 타입 float32 으로 변경 (행렬로)
       mask[mask != 2.0] = 1.0  # 이미지 내 2.0이 아닌 값 (배경과 동물의 경계와 동물) 은 1 으로 
       mask[mask == 2.0] = 0.0  # 나머지는 0.0 으로 
       mask = torch.tensor(mask) # 넘파이 행렬을 토치 텐서로 
       return mask
   
    def __getitem__(self, i) : # i 번째 데이터와 정답을 반환
        if self.train :        # 학습용 데이터  
            X_train = Image.open(self.X_train[i])
            X_train = self.transforms(X_train)
            Y_train = Image.open(self.Y_train[i])
            Y_train = self.preprocess_mask(Y_train)
            
            return X_train , Y_train
        
        else :   # 평가용 데이터 
            X_test = Image.open(self.X_test[i])
            X_test = self.transforms(X_test)
            Y_test = Image.open(self.Y_test[i])
            Y_test = self.transforms(Y_test)
            
            return X_test, Y_test
        
# UNet 정의하기 
import torch.nn as nn
class UNet(nn.Module):
   def __init__(self):
       super(UNet, self).__init__()   # U-Net의 인코더에 사용되는 은닉층
       self.enc1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # input 값에 2d convolution 연산을 적용
       self.enc1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  
       self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 맥스 풀링으로 downsampling

       self.enc2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
       self.enc2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
       self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

       self.enc3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
       self.enc3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
       self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

       self.enc4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
       self.enc4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
       self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

       self.enc5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
       self.enc5_2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
       
       # 디코더에 사용되는 은닉층
       self.upsample4 = nn.ConvTranspose2d(512, 512, 2, stride=2) #  ConvTranspose2d : Convolution 과정의 역을 하는 느낌인데 역행렬이 아니라 전치행렬을 곱함
       self.dec4_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
       self.dec4_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

       self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
       self.dec3_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
       self.dec3_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

       self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
       self.dec2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
       self.dec2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

       self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
       self.dec1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
       self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
       self.dec1_3 = nn.Conv2d(64, 1, kernel_size=1)



       self.relu = nn.ReLU()        # 합성곱과 업샘플링층의 활성화 함수
       self.sigmoid = nn.Sigmoid()  # 출력층의 활성화함수
       
   def forward(self, x):    # forward 연산 실행
       x = self.enc1_1(x)   # 인코더 층 차례로 연산 
       x = self.relu(x)
       e1 = self.enc1_2(x)  # 디코더에서 사용하기 위해 따로 변수를 지정
       e1 = self.relu(e1)   # 합성곱층의 출력의 활성화
       x = self.pool1(e1)

       x = self.enc2_1(x)
       x = self.relu(x)
       e2 = self.enc2_2(x)
       e2 = self.relu(e2)
       x = self.pool2(e2)

       x = self.enc3_1(x)
       x = self.relu(x)
       e3 = self.enc3_2(x)
       e3 = self.relu(e3)
       x = self.pool3(e3)

       x = self.enc4_1(x)
       x = self.relu(x)
       e4 = self.enc4_2(x)
       e4 = self.relu(e4)
       x = self.pool4(e4)

       x = self.enc5_1(x)
       x = self.relu(x)
       x = self.enc5_2(x)
       x = self.relu(x)
       x = self.upsample4(x)

       # 인코더의 출력과 업샘플링된 이미지를 함침
       x = torch.cat([x, e4], dim=1) # cat  : Tensor 합치기 
       x = self.dec4_1(x)
       x = self.relu(x)
       x = self.dec4_2(x)
       x = self.relu(x)

       x = self.upsample3(x)
       x = torch.cat([x, e3], dim=1)
       x = self.dec3_1(x)
       x = self.relu(x)
       x = self.dec3_2(x)
       x = self.relu(x)

       x = self.upsample2(x)
       x = torch.cat([x, e2], dim=1)
       x = self.dec2_1(x)
       x = self.relu(x)
       x = self.dec2_2(x)
       x = self.relu(x)

       x = self.upsample1(x)
       x = torch.cat([x, e1], dim=1)
       x = self.dec1_1(x)
       x = self.relu(x)
       x = self.dec1_2(x)
       x = self.relu(x)
       x = self.dec1_3(x)

       x = torch.squeeze(x)  # 흑백 이미지를 그리기 위해 채널을 없앰

       return x
        
# 데이터 전처리 정의
import tqdm

from torchvision.transforms import Compose # 데이터를 여러 단계로 변환해야 하는 경우, Compose를 통해 여러 단계로 묶음
from torchvision.transforms import ToTensor, Resize # tensor로 변환, 사이즈 변환
from torch.optim.adam import Adam # 옵티마이저 
from torch.utils.data.dataloader import DataLoader #데이터 불러올떄 사용 

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = Compose([
   Resize((128, 128)), # 사이즈 변환 
   ToTensor()          # 텐서로 변환
])

# 데이터 불러오기
# 학습 데이터
train_set = Pets(path_to_img=path_to_image,     
                 path_to_anno=path_to_annotation,
                 transforms=transform)

# 평가용 데이터
test_set = Pets(path_to_img=path_to_image,
                path_to_anno=path_to_annotation, 
                transforms=transform, 
                train=False)

train_loader = DataLoader(train_set , batch_size=32, shuffle=True ) # 여기서 에러 ~! 
test_loader = DataLoader(test_set)

# 학습에 필요한 요소 정의
# 모델 정의
model = UNet().to(device) # 디바이스에 모델 전달 
learning_rate = 0.0001    # 학습률 정의
optim = Adam(params=model.parameters(), lr=learning_rate) # 최적화 정의

for epoch in range(10): # 학습 루프 정의
    iterator = tqdm.tqdm(train_loader)

    for data, label in iterator:
       optim.zero_grad()  # 이전 루프의 기울기 초기화

       preds = model(data.to(device))  # 모델의 예측값 출력
       loss = nn.BCEWithLogitsLoss()(  # nn.BCELoss 에 Sigmoid 함수가 포함된 형태 (BCELoss : Binary Cross Entropy Loss)
           preds, 
           label.type(torch.FloatTensor).to(device))  # 손실 계산
       loss.backward()  # 오차 역전파

       optim.step()  # 최적화

       iterator.set_description(f"epoch{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "./UNet.pth")  # 모델 가중치 저장    

# 모델 성능 평가
model.load_state_dict(torch.load("./UNet.pth", map_location="cpu")) # 모델 불러오기 
data, label = test_set[1]
pred = model(torch.unsqueeze(data.to(device), dim=0))>0.5  # 픽셀을 이진 분류함

with torch.no_grad():     # autograd를 끔
   plt.subplot(1, 2, 1 )  # 그래프 그리기
   plt.title("Predicted") # 제목 붙이기 
   plt.imshow(pred)       # 그래프 그리기
   plt.subplot(1, 2, 2)  
   plt.title("Real")      
   plt.imshow(label)      
   plt.show()        
        
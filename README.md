## Foot Ankle Auto Diagnosis based on Deeplearning
Establishment of multimodality datasets for diagnosis and treatment monitoring of foot and ankle diseases. Development of an artificial intelligence model for predicting foot disease through an automatic measurement system through data collection using musculoskeletal images as the diagnosis of the ankle joint is difficult and complicated in diagnosing foot diseases

## Demo Video
https://user-images.githubusercontent.com/100255173/156266423-fd254b86-9cdc-4aa5-87b6-c2ca4e4bcf0b.mp4

## X-Ray Auto Segmentation
![image01](https://user-images.githubusercontent.com/100255173/156273017-ba91890b-a486-44db-b4e8-9c5680f658df.png)
X-Ray Auto Segmentation의 상세한 구조는 위의 그림과 같다. 먼저 백본 네트워크로 Feature Pyramid Networks를 사용한다. 이는 다양한 스케일에서도 semantic 정보를 잘 담고 있는 feature map 뽑아낼 수 있는 특징을 가지고 있다. 이 때문에 작은 객체에서도 탐지할 수 있다. 이어서 이미지에서 직접적으로 instance의 segment map를 찾아내기 위해 object의 크기와 위치에 따라 나누어 병렬도 동시 예측을 진행한다. 주어진 이미지를 SxS의 grid로 나누고, 예측해야 하는 object의 중심점이 grid cell에 들어가면 해당 object의 segment를 예측할 준비를 한다. 예를 들어 하나의 object의 중심이 grid cell에 들어가게 되면, 그 cell은 해당 object의 category와 instance mask를 예측해야 한다. grid 관점에서 network를 보면 각각의 grid cell은kernel branch에서 D개의 category 확률을 예측하고, feature branch에서 그 object의 instance segment mask를 찾아낸다. 그리고 이는 동시에 진행되기 위해 category prediction이 진행되는 동안 같이 이미지 전체에 대한 객체의 특징점을 미리 뽑아 둔다. 마직막으로 category 분류된 해당 특징만을 마스크를 만들어 낸다. 
기존의 모델과 달라진 점은 vanilla head방식을 decoupled head방식로 변경한 것이다. 기존의 모델은 백본에서 convolution 연산을 거쳐 채널축의 깊이가 S^2인 텐서를 만들어 낸다. 이는 같은 object를 중복해서 잡는다던가 의미 없이 아무것도 잡지 못하는 경우가 발생하기도 하므로 S^2를 연산한다는 것은 매우 무겁다. 그래서 head 부분에서 x축과 y축을 따로 예측하는 decoupled head를 적용한 것이다. 그렇게 되면 S^2였던 것을 S로 다운시킬 수 있고 output space를 H×W×2S로 줄일 수 있다. 

## X-Ray Auto Line Detection
![image02](https://user-images.githubusercontent.com/100255173/156273197-645f83e5-6d42-4054-be83-612f5ed2de99.png)
기존의 Line Detection 모델은 얼굴인식의 윤곽 특징 추출 또는 자율 주행을 위한 차선 인식 등과 같이 입력되는 이미지의 edge가 분명한 경우에만 학습할 수 있다. 
하지만 족부질환 예측을 위한 X-Ray의 뼈의 각도를 측정하기 위해서는 뼈 내부의 non-edge 상태에서 중심부의 Line을 탐지해야 한다. 이를 위해서 개발한 Auto Line Detection은 Unet과 CNN(Convolution Neural Network)을 결합한 Global Universal U-Net을 사용한다. U-net의 경우 Constracting path에서 풀링 전에 Feature map을 뽑아내서 Upsampling 시 결합을 통해 해상도가 높은 수준을 유지하는 기능이 있다. 이를 이용해 전반적이고 추상적인 특징을 추출하고 CNN에서 이차적으로 세밀한 부분의 특징 추출하여 edge가 없는 영역에서도 특정 Landmark를 탐지할 수 있다. 이후 두 개의 Landmark를 하나의 클래스로 설정하여 Line을 탐지한다. 

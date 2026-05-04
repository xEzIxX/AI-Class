## 딥러닝 모델 구조

CNN, RNN, LSTM은 **모두 딥러닝에서 사용되는 신경망 구조**이다.  

| 구분       | CNN             | RNN           | LSTM              |
| -------- | --------------- | ------------- | ----------------- |
| 주 사용 데이터 | 이미지, 영상         | 문장, 음성, 시계열   | 긴 문장, 긴 시계열       |
| 핵심 역할    | 공간적 특징 추출       | 이전 정보 기억      | 중요한 정보 오래 기억      |

---

### 1. CNN

**CNN: Convolutional Neural Network, 합성곱 신경망**

CNN은 이미지와 같이 **가로와 세로의 공간 구조가 중요한 데이터**를 처리하기 위해 설계된 신경망이다.
일반 신경망처럼 이미지를 단순히 1차원으로 펼쳐서 처리하면 픽셀 간의 위치 관계가 약해질 수 있다. CNN은 이를 보완하기 위해 **Filter를 이용한 Convolution 연산**으로 이미지 안의 특징을 감지한다.

Convolution 연산에서는 작은 Filter가 이미지 위를 이동하면서 이미지의 일부 영역과 곱해지고 더해진다. 이 과정을 통해 Feature Map이 만들어지며, CNN은 선, 모서리, 색 변화, 질감, 눈, 귀, 얼굴 형태와 같은 특징을 단계적으로 찾아낼 수 있다.

또한 CNN은 하나의 Filter를 이미지 전체에 반복해서 사용하기 때문에 **Parameter Sharing**이 가능하다. 이를 통해 Fully Connected Layer만 사용하는 방식보다 학습해야 하는 Parameter 수를 줄일 수 있다. 

<img width="996" height="431" alt="image" src="https://github.com/user-attachments/assets/db4179c1-062a-45ff-97dd-4f3532886269" />


#### 구조

* **Convolution Layer**: 입력 데이터에 Filter를 적용하여 Feature Map을 만든다.
* **Activation Function**: Convolution 결과에 비선형성을 추가하여 복잡한 패턴을 학습할 수 있게 한다.
* **Pooling Layer**: Feature Map의 크기를 줄여 중요한 정보를 요약하고 계산량을 줄인다.
* **Flatten**: 다차원 Feature Map을 1차원 벡터로 변환한다.
* **Fully Connected Layer**: 앞에서 추출한 특징을 바탕으로 최종 결과를 판단한다.

#### 적용 분야

* 이미지 및 비디오 인식
* 객체 탐지
* 이미지 분류

#### 장점

* 이미지처럼 공간 구조가 중요한 데이터를 처리하는 데 효과적이다.
* Filter를 통해 이미지의 특징을 자동으로 추출할 수 있어 전처리 부담이 줄어든다.
* Parameter Sharing을 통해 Fully Connected 방식보다 학습해야 하는 Parameter 수를 줄일 수 있다.

#### 단점

* 계산량이 많고, 학습에 많은 메모리가 필요할 수 있다.
* 학습을 위해 많은 양의 Labeling Data가 필요하다.
* 데이터가 부족하면 Overfitting이 발생할 수 있다.

> #### 용어 정리
> * **Convolution**: Filter를 이미지 위에 움직이며 특징을 추출하는 연산
> * **Filter / Kernel**: 이미지에서 특정 특징을 찾기 위한 작은 행렬
> * **Feature Map**: Filter를 적용한 결과로 만들어지는 특징 지도
> * **Activation Function**: 신경망이 복잡한 패턴을 학습할 수 있도록 비선형성을 추가하는 함수
> * **Pooling**: Feature Map의 크기를 줄이고 중요한 정보만 요약하는 과정
> * **Flatten**: 다차원 데이터를 1차원 벡터로 펼치는 과정
> * **Parameter Sharing**: 하나의 Filter를 여러 위치에 반복해서 사용하여 같은 가중치를 공유하는 방식

---

### 2. RNN

**RNN: Recurrent Neural Network, 순환 신경망**

RNN은 문장, 음성, 시계열 데이터처럼 **순서가 중요한 데이터**를 처리하기 위해 설계된 신경망이다.
일반 신경망은 각 입력을 독립적으로 처리하지만, RNN은 **이전 단계의 정보를 기억한 상태에서 현재 입력을 처리**한다.

예를 들어 “I am going to”라는 문장이 입력되면, RNN은 앞의 단어 흐름을 바탕으로 다음 단어가 “school” 또는 “market”이 될 수 있음을 예측할 수 있다. 즉, RNN은 과거의 맥락을 활용하여 다음 결과를 예측하는 데 적합하다.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/f74ae909-9342-4ff6-bcd7-b29d7fe90a00" />


#### 구조

* **Input Layer**: 텍스트, 음성, 시계열 데이터처럼 순서가 있는 입력을 한 단계씩 받는다.
* **Hidden Layer**: 현재 입력과 이전 Hidden State를 함께 사용하여 새로운 Hidden State를 만든다.
* **Recurrent Connection**: 이전 단계의 정보가 다음 단계로 전달되도록 연결된 구조이다.
* **Hidden State**: 이전 입력에서 얻은 정보를 저장하고 다음 단계로 전달하는 값이다.
* **Output Layer**: Hidden State를 바탕으로 최종 예측 결과를 출력한다.

#### 적용 분야

* 시계열 예측
* 자연어 처리, NLP
* 음성 인식
* 기계 번역
* 다음 단어 예측

#### 장점

* 순서가 있는 데이터를 처리하는 데 효과적이다.
* 이전 정보를 활용하여 현재 입력을 해석할 수 있다.
* 입력 데이터의 길이가 일정하지 않아도 처리할 수 있다.
* 각 시간 단계에서 같은 가중치를 사용하는 **Weight Sharing** 구조를 가진다.

#### 단점

* 데이터의 길이가 길어질수록 오래전 정보를 기억하기 어렵다.
* 학습 과정에서 앞쪽 정보가 뒤로 갈수록 약해져서 오래전 정보를 잘 학습하지 못하는 Vanishing Gradient Problem에 취약하다.
* 장기 의존성을 학습하기 어렵기 때문에 LSTM이나 GRU와 같은 발전된 구조로 보완하기도 한다.

> #### 용어 정리
> * **Sequential Data**: 순서가 중요한 데이터. 예를 들어 문장, 음성, 주가, 날씨 데이터 등이 있다.
> * **NLP**: Natural Language Processing, 자연어 처리. 사람이 사용하는 언어를 컴퓨터가 이해하고 처리하는 기술
> * **GRU**: Gated Recurrent Unit. LSTM과 비슷하게 장기 의존성 문제를 완화하기 위해 사용되는 RNN 계열 구조

---

### 3. LSTM

**LSTM: Long Short-Term Memory, 장단기 기억 신경망**

LSTM은 RNN의 한 종류로, RNN이 가진 **오래전 정보를 잘 기억하지 못하는 문제**를 보완하기 위해 만들어졌다.
기본 RNN은 데이터의 길이가 길어질수록 앞부분의 정보를 잊기 쉽고, 학습 과정에서 **Vanishing Gradient Problem**이나 **Exploding Gradient Problem**이 발생할 수 있다.

LSTM은 이를 해결하기 위해 **Memory Cell**과 **Gate** 구조를 사용한다.
중요한 정보는 오래 유지하고, 필요 없는 정보는 버리면서 긴 순서 데이터도 효과적으로 처리할 수 있다.

예를 들어 긴 문장에서 앞부분에 나온 중요한 단어가 뒤쪽 문장의 의미를 이해하는 데 필요할 때, LSTM은 그 정보를 더 오래 기억할 수 있다.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/a86290e9-432b-480d-a1d9-5c32f8e94380" />


#### 구조

* **Memory Cell**: 시간이 지나도 중요한 정보를 저장하고 업데이트하는 역할을 한다.
* **Cell State**: 장기 기억처럼 정보를 다음 시간 단계로 전달하는 흐름이다.
* **Hidden State**: 현재 시점의 출력이자 다음 단계로 전달되는 단기 기억 정보이다.
* **Forget Gate**: 기존 정보 중 어떤 정보를 버릴지 결정한다.
* **Input Gate**: 새로 들어온 정보 중 어떤 정보를 저장할지 결정한다.
* **Output Gate**: Memory Cell의 정보 중 어떤 정보를 다음 단계의 Hidden State로 내보낼지 결정한다.

#### LSTM 적용 분야

* 장기 시계열 예측
* 자연어 처리, NLP
* 기계 번역
* 음성 인식
* 음성 합성
* 이상 탐지

#### 장점

* RNN보다 오래전 정보를 더 잘 기억할 수 있다.
* Long-Term Dependency를 학습하는 데 효과적이다.
* Vanishing Gradient Problem을 완화할 수 있다.
* 필요한 정보는 유지하고 불필요한 정보는 제거할 수 있다.

#### 단점

* 기본 RNN보다 구조가 복잡하다.
* 계산량이 많다.
* Hyperparameter를 신중하게 조정해야 한다.
* 데이터나 문제에 따라 학습 시간이 길어질 수 있다.

> #### 용어 정리
> * **Long-Term Dependency**: 오래전에 나온 정보가 뒤쪽 결과에 영향을 주는 관계
> * **Vanishing Gradient Problem**: 학습 과정에서 기울기가 점점 작아져 오래전 정보를 잘 학습하지 못하는 문제
> * **Exploding Gradient Problem**: 학습 과정에서 기울기가 너무 커져 학습이 불안정해지는 문제
> * **Hyperparameter**: 모델이 학습하기 전에 사람이 미리 정해야 하는 설정값. 예를 들어 Learning Rate, Hidden Layer Size, Batch Size 등이 있다.

## 출처

- [Medium, ANN vs CNN vs RNN vs LSTM: Understanding the Differences in Neural Networks](https://medium.com/@hassaanidrees7/ann-vs-cnn-vs-rnn-vs-lstm-understanding-the-differences-in-neural-networks-94486cbb6d5a)
- [Introduction to Convolution Neural Network](https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/)
- [Recurrent Neural Networks in R](https://www.geeksforgeeks.org/r-language/recurrent-neural-networks-in-r/)
- [What is LSTM - Long Short Term Memory?](https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/)
- [인공지능(AI) 딥러닝과 알고리즘(LSTM)에 대해 알아보자](https://rkdtmdqja98.tistory.com/16)

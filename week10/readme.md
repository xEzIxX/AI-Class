# CNN 모델 구조 변경 실험 정리

## 실험 1: 기존 `mnist_cnn.ipynb` 파일에서 필터 크기를 변경

### [변경 사항]

- 첫 번째 Conv2D layer의 필터 크기를 `3x3`에서 `5x5`로 변경
- 첫 번째 Conv2D의 output shape는 `(26, 26, 32)`에서 `(24, 24, 32)`로 감소
- 첫 번째 Conv2D layer의 Parameter 수는 `320개`에서 `832개`로 증가
- 전체 Parameter 수는 `93,322개`에서 `93,834개`로 증가

### [변경 이유 및 해석]

첫 번째 Conv2D의 output shape가 감소한 이유는 Keras의 Conv2D가 기본적으로 `padding='valid'`를 사용하기 때문이다. 이는 이미지 바깥쪽에 빈 칸을 추가하지 않고, 실제 이미지 안에서만 필터를 움직인다는 의미이다.

따라서 필터 크기가 `3x3`에서 `5x5`로 커지면 필터가 이미지 안에서 움직일 수 있는 위치가 줄어들고, output shape도 작아진다.

또한 Parameter 수가 증가한 이유는 Conv2D layer의 Parameter 수가 필터 크기, 입력 채널 수, 필터 개수에 의해 결정되기 때문이다.

- 변경 전: `(3 × 3 × 1 + 1) × 32 = 320`
- 변경 후: `(5 × 5 × 1 + 1) × 32 = 832`

즉, 필터 크기를 키우면 더 넓은 영역의 특징을 학습할 수 있지만, 그만큼 학습해야 하는 Parameter 수와 연산량도 증가한다.

---

## 실험 2: 기존 `mnist_cnn.ipynb` 파일에서 Conv layer 수를 변경

### [변경 사항]

- 기존 세 번째 Conv2D layer 뒤에 `Conv2D(128, (3, 3), activation='relu')`를 추가
- Conv2D layer 수가 `3개`에서 `4개`로 증가
- 마지막 Conv2D의 output shape가 `(3, 3, 64)`에서 `(1, 1, 128)`로 변경
- Flatten layer의 output shape가 `(576)`에서 `(128)`로 감소
- 추가된 Conv2D layer의 Parameter 수는 `73,856개`
- Dense layer의 Parameter 수는 `36,928개`에서 `8,256개`로 감소
- 전체 Parameter 수는 `93,322개`에서 `138,506개`로 증가

### [변경 이유 및 해석]

Conv layer를 추가하면 모델이 더 깊어져 이미지의 복잡한 특징을 추가로 학습할 수 있다. 기존 모델은 Conv2D layer가 3개였지만, 실험 2에서는 Conv2D layer를 하나 더 추가하여 총 4개로 늘렸다.

추가된 Conv2D layer는 기존 `(3, 3, 64)` 크기의 feature map에 `3x3` 필터를 적용한다. Conv2D의 기본 padding이 `valid`이기 때문에 output shape는 `(1, 1, 128)`로 줄어든다.

추가된 Conv2D layer의 Parameter 수는 다음과 같이 계산된다.

- `(3 × 3 × 64 + 1) × 128 = 73,856`

Flatten 결과가 `576`에서 `128`로 줄어들었기 때문에 Dense layer의 Parameter 수는 오히려 감소하였다.

- 변경 전: `576 × 64 + 64 = 36,928`
- 변경 후: `128 × 64 + 64 = 8,256`

하지만 새로 추가된 Conv2D layer의 Parameter 수가 크기 때문에 전체 Parameter 수는 증가하였다.

즉, Conv layer를 추가하면 모델의 표현력은 증가하지만, 전체적으로 학습해야 하는 Parameter 수와 연산량도 증가한다.

---

## 실험 3: 기존 `mnist_cnn.ipynb` 파일에서 Dense layer를 변경

### [변경 사항]

- 기존 `Dense(64, activation='relu')`를 `Dense(128, activation='relu')`로 변경
- Dense layer의 output shape가 `(None, 64)`에서 `(None, 128)`로 증가
- Conv2D layer들의 output shape는 기존과 동일하게 유지
- Flatten layer의 output shape도 `(None, 576)`으로 동일
- Dense layer의 Parameter 수는 `36,928개`에서 `73,856개`로 증가
- 마지막 출력층 Dense layer의 Parameter 수는 `650개`에서 `1,290개`로 증가
- 전체 Parameter 수는 `93,322개`에서 `130,890개`로 증가

### [변경 이유 및 해석]

실험 3에서는 Conv2D layer 구조를 변경하지 않고, Flatten 이후의 Dense layer만 변경하였다. 따라서 이미지 특징을 추출하는 Conv2D layer들의 output shape는 기존과 동일하게 유지되었다.

Flatten layer의 output shape도 `(None, 576)`으로 동일하므로 Dense layer에 입력되는 값의 개수는 변하지 않았다. 하지만 Dense layer의 노드 수를 `64개`에서 `128개`로 늘렸기 때문에 Dense layer에서 학습해야 하는 Parameter 수가 증가하였다.

Dense layer의 Parameter 수는 다음과 같이 계산된다.

- 변경 전: `576 × 64 + 64 = 36,928`
- 변경 후: `576 × 128 + 128 = 73,856`

또한 마지막 출력층도 이전 layer의 노드 수가 64개에서 128개로 증가하면서 Parameter 수가 함께 증가하였다.

- 변경 전: `64 × 10 + 10 = 650`
- 변경 후: `128 × 10 + 10 = 1,290`

즉, 이번 실험에서 전체 Parameter 수가 증가한 이유는 Conv2D 구조 변화가 아니라 Dense layer의 노드 수 증가 때문이다.

Dense layer의 크기를 늘리면 더 많은 특징 조합을 학습할 수 있지만, 그만큼 Parameter 수와 연산량도 증가한다. 따라서 정확도 향상이 크지 않다면 모델이 불필요하게 복잡해진 것으로 해석할 수 있다.

---

## 전체 실험 요약

| 실험 | 변경 내용 | 주요 변화 |
|---|---|---|
| 실험 1 | 첫 번째 Conv2D 필터 크기 `3x3 → 5x5` | output shape 감소, Parameter 수 증가 |
| 실험 2 | Conv2D layer 1개 추가 | 모델 깊이 증가, 전체 Parameter 수 증가 |
| 실험 3 | Dense layer 노드 수 `64 → 128` | Dense Parameter 수 증가, 전체 Parameter 수 증가 |

## 종합 해석

세 실험 모두 기존 CNN 모델의 구조를 변경하여 output shape와 Parameter 수가 어떻게 달라지는지 확인하기 위한 실험이다.

실험 1에서는 필터 크기를 키우면서 더 넓은 이미지 영역을 한 번에 학습할 수 있게 되었지만, output shape는 감소하고 Parameter 수는 증가하였다.

실험 2에서는 Conv layer를 추가하여 모델을 더 깊게 만들었다. 이를 통해 더 복잡한 특징을 학습할 수 있지만, 추가된 Conv2D layer로 인해 전체 Parameter 수가 증가하였다.

실험 3에서는 Dense layer의 노드 수를 늘려 Flatten 이후의 특징 조합 능력을 높였다. 하지만 Conv2D 구조는 그대로였기 때문에 이미지 특징 추출 과정은 변하지 않았고, Dense layer와 출력층의 Parameter 수만 증가하였다.

따라서 CNN 모델에서는 필터 크기, Conv layer 수, Dense layer 크기를 변경하면 모델의 표현력은 증가할 수 있지만, 동시에 Parameter 수와 연산량도 증가한다. 그러므로 단순히 모델을 크게 만드는 것보다 accuracy, loss, Parameter 수를 함께 비교하여 적절한 구조를 선택하는 것이 중요하다.

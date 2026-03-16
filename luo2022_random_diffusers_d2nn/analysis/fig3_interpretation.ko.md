# Fig. 3 해석 가이드

## 권장 캡션 문안

### 영문 캡션 보강 문장

Add the following sentence after the first sentence that introduces the measured periods in Fig. 3(a,b):

> Here, the Resolution Test Target Period denotes the ground-truth period of the input binary resolution target, whereas the Measured Grating Period denotes the period estimated from the reconstructed output image at the D2NN output plane.

### 한글 설명 버전

여기서 `Resolution Test Target Period`는 입력 해상도 타깃이 실제로 갖는 기준 주기이고, `Measured Grating Period`는 디퓨저와 D2NN을 통과한 뒤 출력면에서 재구성된 영상으로부터 추정한 주기이다.

## 이 설명을 어디에 넣으면 좋은가

- 가장 좋은 위치는 Fig. 3 캡션에서 `measured periods`를 처음 언급한 직후이다.
- 대안으로는 Fig. 3를 해설하는 본문 첫 문단의 첫 두 문장 뒤에 넣을 수 있다.
- 캡션에는 용어 정의를 짧게 넣고, 본문에는 왜 두 값이 같아야 좋은 일반화인지 한 문장 더 설명하는 구성이 가장 읽기 좋다.

## 용어 차이

### Resolution Test Target Period

- 입력 타깃 자체의 물리적 주기이다.
- x축 값이다.
- 실험자가 이미 알고 있는 정답값이다.

### Measured Grating Period

- 출력면에서 재구성된 intensity profile로부터 추정한 주기이다.
- y축 값이다.
- 복원 성능이 좋을수록 x축의 true period와 더 가깝게 나온다.

## Fig. 3를 어떻게 읽어야 하나

Fig. 3는 "MNIST 숫자만 보고 학습한 네트워크가, 학습 때 보지 못한 선형 해상도 타깃에도 일반화되는가?"를 보는 그림이다. 따라서 이 그림의 핵심은 숫자 분류 정확도가 아니라, **보지 못한 구조적 패턴의 주기를 얼마나 정확히 복원하느냐**이다.

### 패널 (a): Last n Diffusers in Training

- 각 네트워크가 마지막 학습 epoch에서 실제로 사용했던 diffuser들로 테스트한 결과이다.
- 즉, training distribution에 가장 가까운 조건이다.
- 막대가 초록색 `True Period` 선과 잘 맞으면, 적어도 학습에 사용된 diffuser 조건에서는 주기 복원이 정확하다는 뜻이다.

### 패널 (b): 20 New Diffusers

- 학습에 한 번도 쓰지 않은 새로운 random diffuser 20개로 테스트한 결과이다.
- 이 패널이 진짜 일반화 성능이다.
- 막대가 여전히 초록색 기준선 근처에 머물면, 네트워크가 특정 diffuser를 외운 것이 아니라 diffuser family 전체에 대해 강건한 복원 규칙을 학습했다고 해석할 수 있다.

## 색 막대는 무엇을 의미하나

- 파란색: `n = 1`
- 주황색: `n = 10`
- 노란색: `n = 15`
- 보라색: `n = 20`
- 초록 점선: 해당 입력 타깃의 true period

여기서 `n`은 각 학습 epoch에서 사용한 diffuser 개수이다. `n`이 커질수록 학습 중 더 다양한 diffuser를 보게 되므로, 보통 새로운 diffuser에 대한 일반화가 좋아질 것으로 기대한다.

## 이 그림의 핵심 해석 포인트

1. 막대 높이가 초록 점선과 가까운가

가까울수록 출력 재구성으로부터 추정한 주기가 실제 입력 주기와 잘 맞는다는 뜻이다. 즉, 복원 정확도가 높다.

2. 패널 (a)와 (b)의 차이가 작은가

작을수록 known diffuser와 unseen diffuser 사이의 generalization gap이 작다는 뜻이다.

3. `n`이 증가할수록 panel (b)가 안정되는가

`n=1`보다 `n=10,15,20`이 새 diffuser에서도 비슷한 주기를 재현하면, 학습 중 diffuser 다양성을 늘리는 것이 일반화에 도움이 된다고 해석할 수 있다.

4. 저주기 영역에서 오차가 더 큰가

작은 period는 더 높은 공간주파수 성분을 포함하므로 복원이 더 어려울 수 있다. 따라서 7.2 mm, 8.4 mm 쪽에서 오차가 커지면 시스템의 해상 한계를 시사한다.

## 본문 해설에 바로 넣기 좋은 문장

### 짧은 버전

The x-axis reports the ground-truth period of each input resolution target, while the y-axis reports the period measured from the reconstructed output image. Therefore, bars that remain close to the green true-period markers indicate accurate period recovery and successful generalization to unseen line-based objects.

### 조금 더 해설적인 버전

Although these binary resolution targets were never included in MNIST-based training, the measured grating periods at the output plane remain close to the true target periods for both the last training diffusers and previously unseen random diffusers. This indicates that the trained diffractive networks generalize beyond the digit manifold and preserve the spatial frequency content of unseen line-based objects.

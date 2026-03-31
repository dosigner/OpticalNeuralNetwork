# PRD: GPU-Accelerated Free-Space Optical Beam Propagation Simulator
## Through Atmospheric Turbulence (Split-Step Angular-Spectrum Method)

**Version**: 1.0  
**Author**: DJ (ADD Optical Network Research Section)  
**Date**: 2026-03-19  
**Platform**: PyTorch + CUDA (A100 40GB), complex128 precision  

---

## 1. 목적 (Objective)

Constant $C_n^2$ 환경에서 콜리메이티드 가우시안 빔의 대기 난류 전파를 시뮬레이션하여,
수신단(observation plane)에서의 **complex optical field**와 **irradiance**를 
Monte-Carlo 방식으로 다수 생성하는 end-to-end 시뮬레이터.

진공(vacuum) 전파 결과와 난류(turbulence) 전파 결과를 모두 출력하며,
물리적 타당성은 structure function 및 coherence factor 기반 자동 검증으로 점검한다.

---

## 2. 확정된 설계 결정 (Design Decisions)

| 항목 | 결정 | 근거 |
|------|------|------|
| 광원 | 콜리메이티드 가우시안 빔 (waist at transmitter) | 사용자 입력: full-divergence → $w_0$ 역산 |
| 파장 $\lambda$ | 1550 nm 고정 | FSO uplink 표준 파장 |
| 난류 PSD | **Kolmogorov only** ($l_0=0$, $L_0=\infty$) | 검증 우선, Eq (9.16) |
| 격자 설정 | $\delta_n$ 사용자 지정, $\delta_1$ 자동 계산 | Constraint 1 기반 역산 |
| Receiver | 관측 computational window ($D_\text{ROI}$) + optional analysis aperture ($D_\text{ap}$) | sampling은 $D_2 = D_\text{ROI}$, coherence 검증은 aperture mask 사용 |
| Phase screen r0 분배 | SciPy constrained optimization | Listing 9.5: $r_{0,sw}$와 $\sigma_\chi^2$ 동시 매칭 |
| Phase screen 생성 | **FT + Subharmonic** (3-level) | Listing 9.3: 저주파 정확도 보장 |
| 검증 | 자동 (structure function + masked coherence factor) | Eq (9.44) + Eq (9.32) 기반 reference check |
| 출력 | Complex field + irradiance (.pt 저장) | 각 realization별 |
| 연산 | PyTorch, **complex128**, A100 GPU | FFT 정밀도 확보 |

---

## 3. 사용자 입력 파라미터 (User Input Parameters)

```python
@dataclass
class SimulationConfig:
    # --- 전파 기하 ---
    Dz: float           # 전파 거리 [m] (예: 50e3)
    Cn2: float          # 굴절률 구조 상수 [m^{-2/3}] (예: 1e-16), constant along path
    
    # --- 광원 ---
    theta_div: float    # Full-angle far-field divergence [rad] (예: 1e-3)
                        # → w0 = λ / (π * theta_div / 2)  [Eq. A1 아래 참조]
    
    # --- 관측면 ---
    D_roi: float              # 관측 computational window 직경 [m] (예: 2.0)
    delta_n: float            # 관측면 격자 간격 [m] (예: 10e-3)
    D_aperture: float | None = None
                              # coherence-factor 검증용 circular analysis aperture [m]
                              # None이면 full-window mask (= D_roi)
    
    # --- 시뮬레이션 ---
    N: int = None       # 격자 점 수 (2의 거듭제곱). None이면 자동 계산
    n_reals: int = 20   # 난류 realization 수
    
    # --- 고정값 (변경 불가) ---
    wvl: float = 1550e-9  # 파장 [m]
    dtype: str = "complex128"  # reference run dtype
```

**자동 계산되는 파라미터:**
- $\delta_1$: Sampling constraint 1 (Eq 9.86)로부터 역산
- $N$: Sampling constraint 2 (Eq 9.87)로부터 하한, 2의 거듭제곱으로 올림
- $n_\text{scr}$: Constraint 4 (Eq 8.24, 9.89-9.90)로부터 최소 평면 수 계산
- 각 screen의 $r_{0,i}$: Constrained optimization (Listing 9.5 방식)

---

## 4. 물리 모델 및 수식 매핑 (Physics → Code Mapping)

### 4.1. 광원 모델: 콜리메이티드 가우시안 빔

사용자가 입력하는 full-angle far-field divergence $\theta_\text{div}$로부터 빔 웨이스트 역산:

$$
\theta_\text{div} = \frac{2\lambda}{\pi w_0}
\quad\Longrightarrow\quad
w_0 = \frac{2\lambda}{\pi \theta_\text{div}}
$$

송신단(source plane, $z=0$)에서의 필드:

$$
U(\mathbf{r}_1) = \exp\!\Bigl(-\frac{|\mathbf{r}_1|^2}{w_0^2}\Bigr)
$$

- 콜리메이티드이므로 wavefront radius of curvature $R = \infty$ (평면 위상).
- 진폭 정규화는 총 파워 기준으로 수행 (선택사항, 상대 비교에는 불필요).

**주의**: PDF의 점광원 모델(sinc function, Listing 9.7)과는 다름.
가우시안 빔은 $R=\infty$이므로, Constraint 3 (Eq 8.22)에서 $(1+\Delta z/R) = 1$로 단순화됨.

$$
\delta_1 - \frac{\lambda\Delta z}{D_1} \;\leq\; \delta_n \;\leq\; \delta_1 + \frac{\lambda\Delta z}{D_1}
\tag{Eq 8.22, R=\infty}
$$

여기서 $D_1$은 source-plane에서 빔이 유의미한 에너지를 갖는 영역의 크기.
가우시안 빔의 경우, $D_1 \approx 4w_0$ ~ $6w_0$ (에너지의 99.97%~99.9999% 포함)로 설정.

**구현 시 결정**: $D_1 = 4w_0$ 사용 (에너지 99.97% 포함, super-Gaussian absorbing boundary가 나머지 처리).

### 4.2. 대기 파라미터 계산

파장 고정: $\lambda = 1550 \times 10^{-9}$ m, $k = 2\pi/\lambda$.

#### 4.2.1. Fried parameter (coherence diameter)

**평면파** (Eq 9.42, constant $C_n^2$):
$$
r_{0,pw} = \left(0.423 \, k^2 \, C_n^2 \, \Delta z\right)^{-3/5}
$$

**구면파** (Eq 9.43, constant $C_n^2$):

적분을 해석적으로 평가:
$$
\int_0^{\Delta z} C_n^2 \left(\frac{z}{\Delta z}\right)^{5/3} dz 
= C_n^2 \cdot \frac{\Delta z}{(5/3)+1} 
= C_n^2 \cdot \frac{3}{8}\Delta z
$$

따라서:
$$
r_{0,sw} = \left(0.423 \, k^2 \, C_n^2 \cdot \frac{3}{8} \Delta z\right)^{-3/5}
\tag{Listing 9.5, line 17}
$$

#### 4.2.2. Rytov variance (log-amplitude variance, 구면파)

Eq (9.64), constant $C_n^2$:
$$
\sigma_{\chi,sw}^2 = 0.563 \, k^{7/6} \int_0^{\Delta z} C_n^2 \, z^{5/6}\left(1-\frac{z}{\Delta z}\right)^{5/6} dz
$$

이 적분은 Beta function으로 해석적으로 평가 가능:
$$
\int_0^{\Delta z} z^{5/6}\left(1-\frac{z}{\Delta z}\right)^{5/6} dz
= \Delta z^{11/6} \cdot B\!\left(\frac{11}{6},\frac{11}{6}\right)
$$

여기서 $B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)$.

**구현**: `scipy.special.beta(11/6, 11/6)` 사용하여 정확히 계산.
또는 수치 적분 (`np.trapz`)으로 Listing 9.5 lines 19-22와 동일하게 구현.

**검증 기준**: $\sigma_\chi^2 < 0.25$이면 weak fluctuation (Rytov 이론 유효).
$\sigma_\chi^2 \gg 0.25$이면 경고 메시지 출력 (시뮬레이션은 계속 수행하되, 이론 검증의 정확도가 떨어질 수 있음을 고지).

#### 4.2.3. 빔 전파에 대한 참고

가우시안 빔은 구면파도 평면파도 아니므로, $r_{0,sw}$와 $\sigma_{\chi,sw}^2$는 
phase screen 분배 최적화 및 검증용 **근사 기준값**으로 사용.
실제 전파 결과는 split-step 시뮬레이션으로 정확하게 계산됨.

### 4.3. Sampling Constraint 분석 및 격자 자동 설정

#### 4.3.1. 난류에 의한 beam spreading 보정 (Sec 9.4)

Eq (9.84)-(9.85)에서 난류에 의한 유효 aperture 확대:
$$
D_1' = D_1 + c\,\frac{\lambda\Delta z}{r_{0,\text{rev}}}
\tag{Eq 9.84}
$$
$$
D_2' = D_\text{ROI} + c\,\frac{\lambda\Delta z}{r_0}
\tag{Eq 9.85}
$$

여기서:
- $c = 2$ (에너지 97% 포함, Listing 9.6 line 2)
- 가우시안 빔(콜리메이티드)의 경우, $r_0 \approx r_{0,sw}$, $r_{0,\text{rev}} \approx r_{0,sw}$ 사용
  (constant $C_n^2$ 수평 경로에서는 순방향/역방향 대칭)

**주의**:
- Sampling constraint에서는 computational observation window로 $D_2 = D_\text{ROI}$ 사용.
- Optional analysis aperture $D_\text{ap}$는 coherence-factor 검증용 mask에만 사용하고,
  constraint 1-3 및 $N$ 계산에는 사용하지 않음.

#### 4.3.2. $\delta_1$ 자동 계산

**Constraint 1** (Eq 9.86):
$$
\delta_n \leq \frac{\lambda\Delta z - D_2'\,\delta_1}{D_1'}
$$

이를 $\delta_1$에 대해 정리:
$$
\delta_1 \leq \frac{\lambda\Delta z - D_1'\,\delta_n}{D_2'}
\tag{★ \delta_1 상한}
$$

**Constraint 3** (Eq 9.88, $R=\infty$):
$$
\delta_1 - \frac{\lambda\Delta z}{D_1} \leq \delta_n \leq \delta_1 + \frac{\lambda\Delta z}{D_1}
$$
$\delta_1$에 대해:
$$
\delta_n - \frac{\lambda\Delta z}{D_1} \leq \delta_1 \leq \delta_n + \frac{\lambda\Delta z}{D_1}
$$

**$\delta_1$ 결정 로직**:
1. Constraint 1에서 $\delta_1$의 상한 계산
2. Constraint 3에서 $\delta_1$의 범위 계산
3. 두 조건의 교집합에서 $\delta_1$ 선택
4. 추가 조건: $D_1 / \delta_1 \geq 5$ (source aperture에 최소 5 샘플)
5. 위 조건을 모두 만족하는 최대 $\delta_1$ 선택 (격자 수 최소화를 위해)

**실패 조건**: 교집합이 공집합이면, $\delta_n$이 너무 크거나 전파 거리가 너무 길어서 해당 격자로는 시뮬레이션 불가. 에러 메시지와 함께 권장 $\delta_n$ 범위 출력.

#### 4.3.3. $N$ 자동 계산

**Constraint 2** (Eq 9.87):
$$
N \geq \frac{D_1'}{2\delta_1} + \frac{D_2'}{2\delta_n} + \frac{\lambda\Delta z}{2\delta_1\delta_n}
$$

계산된 하한을 2의 거듭제곱으로 올림:
$$
N = 2^{\lceil \log_2 N_\text{min} \rceil}
$$

사용자가 `N`을 명시적으로 지정한 경우, constraint 2를 만족하는지 확인하고, 
불만족 시 경고와 함께 최소 $N$ 값을 안내.

#### 4.3.4. Partial propagation 수 계산

**Constraint 4** (Eq 8.24):
$$
\Delta z_\text{max} = \frac{\min(\delta_1, \delta_n)^2 \cdot N}{\lambda}
\tag{Eq 9.89}
$$

$$
n_\text{min} = \left\lceil \frac{\Delta z}{\Delta z_\text{max}} \right\rceil + 1
\tag{Eq 9.90}
$$

$n_\text{scr} = n_\text{min}$ (phase screen 수 = plane 수).

추가 조건: 각 partial propagation의 Rytov number가 0.1 이하 (Listing 9.5, line 37).
이 조건이 $n_\text{min}$보다 더 많은 screen을 요구하면, screen 수를 늘림.

### 4.4. Phase Screen r0 분배 (Constrained Optimization)

**목적**: $n_\text{scr}$개의 phase screen에 대해, 각각의 $r_{0,i}$를 결정하여
bulk 난류의 $r_{0,sw}$와 $\sigma_{\chi,sw}^2$를 동시에 재현.

**수식** (Eq 9.72, 9.74 — $r_{0,i}$ 기반):

Screen 위치: $\alpha_i = z_i / \Delta z$ (균등 간격, $i = 0, 1, \ldots, n_\text{scr}-1$).

$$
r_{0,sw} = \left[\sum_{i=1}^{n} r_{0,i}^{-5/3} \left(\frac{z_i}{\Delta z}\right)^{5/3}\right]^{-3/5}
\tag{Eq 9.72}
$$

$$
\sigma_{\chi,sw}^2 = 1.33\, k^{-5/6}\, \Delta z^{5/6} \sum_{i=1}^{n} r_{0,i}^{-5/3} \left(\frac{z_i}{\Delta z}\right)^{5/6}\left(1-\frac{z_i}{\Delta z}\right)^{5/6}
\tag{Eq 9.74}
$$

**행렬 형태** (Eq 9.75):

$\mathbf{x} = [r_{0,1}^{-5/3}, r_{0,2}^{-5/3}, \ldots, r_{0,n}^{-5/3}]^T$로 정의하면:

$$
\begin{pmatrix}
r_{0,sw}^{-5/3} \\
\sigma_{\chi,sw}^2 \cdot (k/\Delta z)^{5/6} / 1.33
\end{pmatrix}
= \mathbf{A}\,\mathbf{x}
\tag{Eq 9.75 rearranged from Eq 9.74}
$$

여기서:
- $A_{1,i} = \alpha_i^{5/3}$
- $A_{2,i} = \alpha_i^{5/6}(1-\alpha_i)^{5/6}$

**최적화 문제**:
$$
\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2
\quad\text{s.t.}\quad
\mathbf{0} \leq \mathbf{x} \leq \mathbf{x}_\text{max}
$$

- 하한: $x_i \geq 0$ (음의 $r_0$는 비물리적)
- 상한: 각 screen의 Rytov 기여가 0.1 이하 (Listing 9.5, lines 37-39)
  $$x_{\max,i} = \frac{r_\text{max}\,(k/\Delta z)^{5/6}}{1.33\,A_{2,i}}, \quad r_\text{max} = 0.1$$
  $A_{2,i} = 0$인 경우 (첫 번째/마지막 screen): $x_{\max,i}$를 작은 상한으로 고정하여
  endpoint screen이 비현실적으로 강해지지 않도록 함 (예: $50^{-5/3}$, 즉 $r_0 = 50$ m에 해당하는 약한 screen)

**구현**: `scipy.optimize.minimize(method='SLSQP', bounds=...)` 사용.

**초기값**: $x_0 = (n_\text{scr}/3 \cdot r_{0,sw})^{-5/3} \cdot \mathbf{1}$ (Listing 9.5, line 32).

**검증**: 최적화 후 $\mathbf{A}\mathbf{x}$로 복원한 $r_{0,sw}$와 $\sigma_\chi^2$가 
목표값과 1% 이내로 일치하는지 확인. 불일치 시 경고.

### 4.5. Phase Screen 생성 (FT + Subharmonic)

#### 4.5.1. FFT 기반 고주파 screen (Listing 9.2)

Kolmogorov phase PSD (Eq 9.52, ordinary frequency):
$$
\Phi_\phi^K(f) = 0.023\, r_0^{-5/3}\, f^{-11/3}
$$

**주의**: Kolmogorov ($l_0=0$, $L_0=\infty$)에서는 $f=0$에서 발산.
DC 성분($f=0$)을 0으로 설정 (Listing 9.2, line 16): `PSD_phi[N/2, N/2] = 0`.

구현 (Listing 9.2 직역):
```
del_f = 1 / (N * delta)
fx, fy = meshgrid of (-N/2 : N/2-1) * del_f
f = sqrt(fx^2 + fy^2)
PSD_phi = 0.023 * r0^(-5/3) * f^(-11/3)
PSD_phi[center] = 0
cn = (randn(N,N) + 1j*randn(N,N)) * sqrt(PSD_phi) * del_f
phz_hi = real(IFT2(cn, 1))
```

**PyTorch 구현 주의사항**:
- `torch.fft.ifft2` + `torch.fft.fftshift`/`ifftshift` 사용
- 난수 생성은 `torch.randn(..., dtype=torch.float64, device='cuda')`
- PSD 계산은 float64, 결과 phase screen도 float64
- Listing 9.2 line 20과 동일하게 phase synthesis는 `ift2(cn, 1)` 또는
  centered `ifft2(cn) * N^2`에 해당해야 하며, propagation용 `delta_f` scaling을 여기에 넣지 않음

#### 4.5.2. Subharmonic 저주파 보정 (Listing 9.3)

$N_p = 3$개의 subharmonic level, 각각 3×3 주파수 격자:

$$
\phi_{LF}(x,y) = \sum_{p=1}^{3} \sum_{n=-1}^{1} \sum_{m=-1}^{1} c_{n,m}^{(p)} \exp\!\bigl[i2\pi(f_{x_n}x + f_{y_m}y)\bigr]
\tag{Eq 9.81}
$$

각 level $p$의 주파수 간격: $\Delta f_p = 1/(3^p \cdot D)$, 여기서 $D = N\cdot\delta$.

구현 (Listing 9.3 직역):
```
phz_lo = zeros(N, N)
for p in 1, 2, 3:
    del_f_p = 1 / (3^p * D)
    fx_p = [-1, 0, 1] * del_f_p
    fy_p = [-1, 0, 1] * del_f_p
    [fx_grid, fy_grid] = meshgrid(fx_p, fy_p)
    f_grid = sqrt(fx_grid^2 + fy_grid^2)
    PSD_phi_p = 0.023 * r0^(-5/3) * f_grid^(-11/3)
    PSD_phi_p[center] = 0
    cn_p = (randn(3,3) + 1j*randn(3,3)) * sqrt(PSD_phi_p) * del_f_p
    for each (fx_val, fy_val, cn_val) in the 3x3 grid:
        phz_lo += cn_val * exp(i*2*pi*(fx_val*x + fy_val*y))
phz_lo = real(phz_lo) - mean(real(phz_lo))
```

**최종 phase screen**: `phz = phz_lo + phz_hi`

**GPU 구현 참고**: 
- Subharmonic 부분은 3×3 격자에 대한 단순 루프이므로 CPU에서 계산해도 무방
- 또는 broadcasting으로 GPU에서 한 번에 처리 가능

### 4.6. Split-Step Angular-Spectrum 전파 (Listing 9.1)

핵심 전파 알고리즘 (Eq 9.3 = Eq 8.18 + phase screen):

$$
U(\mathbf{r}_n) = \mathcal{Q}\!\left[\frac{m_{n-1}-1}{m_{n-1}\Delta z_{n-1}}, \mathbf{r}_n\right]
\times \prod_{i=1}^{n-1}\left\{
\mathcal{T}[z_i,z_{i+1}]\,
\mathcal{F}^{-1}\!\left[\mathbf{f}_i, \frac{\mathbf{r}_{i+1}}{m_i}\right]
\mathcal{Q}_2\!\left[-\frac{\Delta z_i}{m_i}, \mathbf{f}_i\right]
\mathcal{F}[\mathbf{r}_i, \mathbf{f}_i]\,\frac{1}{m_i}
\right\}
\times\left\{
\mathcal{Q}\!\left[\frac{1-m_1}{\Delta z_1}, \mathbf{r}_1\right]
\mathcal{T}[z_1,z_2]\,U(\mathbf{r}_1)
\right\}
\tag{Eq 9.3}
$$

여기서:
- $\mathcal{Q}[a, \mathbf{r}] = \exp\!\bigl(i\frac{k}{2}a|\mathbf{r}|^2\bigr)$ : 이차 위상 인자
- $\mathcal{Q}_2[a, \mathbf{f}] = \exp\!\bigl(-i\pi^2 \cdot 2a/k \cdot |\mathbf{f}|^2\bigr)$ : 주파수 영역 이차 위상 인자
- $\mathcal{T}[z_i, z_{i+1}] = \exp[-i\phi(\mathbf{r})]$: phase screen (Eq 9.2)
- $\mathcal{F}$: 2D FT, $\mathcal{F}^{-1}$: 2D IFT
- $m_i = \delta_{i+1}/\delta_i$: scaling factor
- Super-Gaussian absorbing boundary가 매 plane에서 적용

**Listing 9.1 직역 (PyTorch pseudocode)**:
```python
# 초기화
z_planes = [0, z1, z2, ..., z_{n-1}]  # n개 plane
Delta_z = diff(z_planes)
alpha = z_planes / z_planes[-1]
delta = (1 - alpha) * delta1 + alpha * deltan   # Eq (8.8)의 일반화
m = delta[1:] / delta[:-1]

# Super-Gaussian absorbing boundary (Listing 9.1, lines 10-12)
nsq = nx^2 + ny^2
w = 0.47 * N
sg = exp(-nsq^8 / w^16)

# 초기 이차 위상 (Listing 9.1, line 25-26)
r1sq = x1^2 + y1^2
Q1 = exp(1j * k/2 * (1 - m[0]) / Delta_z[0] * r1sq)
U = U_source * Q1 * T[0]   # T[0] = phase screen at plane 0

# 반복 전파 (Listing 9.1, lines 27-39)
for i in range(n-1):
    deltaf = 1 / (N * delta[i])
    fX = nx * deltaf
    fY = ny * deltaf
    fsq = fX^2 + fY^2
    Q2 = exp(-1j * pi^2 * 2 * Delta_z[i] / m[i] / k * fsq)
    U = sg * T[i+1] * IFT2(Q2 * FT2(U / m[i]))

# 최종 이차 위상 (Listing 9.1, lines 44-46)
rnsq = xn^2 + yn^2
Q3 = exp(1j * k/2 * (m[-1] - 1) / (m[-1] * Delta_z[-1]) * rnsq)
U_out = Q3 * U
```

**vacuum 전파**: 위 알고리즘에서 $\mathcal{T} = 1$ (모든 phase screen을 1로 설정).
즉 `T[i] = 1`로 두면 진공 전파 (Listing 9.1 설명: "can be used for vacuum propagation if T=1 at every step").

**FT 컨벤션 주의**:
- PDF의 `ft2`/`ift2`는 `delta` 스케일링이 포함된 DFT
- `ft2(g, delta)` = `fftshift(fft2(ifftshift(g))) * delta^2`
- `ift2(G, deltaf)` = `ifftshift(ifft2(fftshift(G))) * (N*deltaf)^2`
- PyTorch에서: `torch.fft.fftshift`, `torch.fft.ifftshift`, `torch.fft.fft2`, `torch.fft.ifft2` 사용
- **complex128** 필수: `torch.complex128` (= `torch.cdouble`)

### 4.7. Vacuum 전파

위 4.6의 알고리즘에서 phase screen 항을 제거:
- `T[i] = 1` for all $i$
- super-Gaussian absorbing boundary는 동일하게 적용

출력: vacuum complex field $U_\text{vac}(\mathbf{r}_n)$ 1개.

### 4.8. 난류 전파 (Monte-Carlo)

각 realization $j = 1, \ldots, n_\text{reals}$에 대해:
1. $n_\text{scr}$개의 독립 phase screen 생성 (각각의 $r_{0,i}$, $\delta_i$ 사용)
2. Split-step 전파 수행
3. Complex field $U_j(\mathbf{r}_n)$ 저장

---

## 5. 자동 검증 (Automated Verification)

### 5.1. Phase Screen Structure Function 검증

**이론** (Eq 9.44, Kolmogorov):
$$
D_\phi^K(r) = 6.88\left(\frac{r}{r_0}\right)^{5/3}
$$

**절차**:
1. Screen plane $i$별로, 동일한 $(r_{0,i}, \delta_i)$를 공유하는 realization 묶음을 수집
2. 각 realization에 대해 2D structure function 계산
3. Plane별 앙상블 평균 계산
4. 각 plane의 이론값 $6.88(r/r_{0,i})^{5/3}$와 1D slice 비교
5. Plane별 상대 오차 $|D_\text{sim}(r) - D_\text{theory}(r)| / D_\text{theory}(r)$ 계산
6. 각 plane의 유효 lag 구간에서 평균 상대 오차 < 15% 판정, 결과는 plane별로 보고

**2D Structure function 계산** (Listing 3.7의 `str_fcn2_ft` 방식):
$$
D_\phi(\Delta\mathbf{r}) = \langle |\phi(\mathbf{r}) - \phi(\mathbf{r}+\Delta\mathbf{r})|^2 \rangle
$$

이것은 autocorrelation으로부터 계산 가능:
$$
D_\phi(\Delta\mathbf{r}) = 2[\sigma_\phi^2 - B_\phi(\Delta\mathbf{r})]
$$
여기서 $B_\phi$는 phase의 autocorrelation, $\sigma_\phi^2 = B_\phi(0)$.

구현: `D = 2 * (B[0,0] - B)`, $B = \text{IFT2}(|\text{FT2}(\phi \cdot \text{mask})|^2) / \text{IFT2}(|\text{FT2}(\text{mask})|^2)$

### 5.2. Coherence Factor 검증

**이론** (Eq 9.32 + Eq 9.44, with $r_0 = r_{0,sw}$ as reference after collimation):
$$
\mu^K(|\Delta\mathbf{r}|) = \exp\!\left[-\frac{1}{2}D(|\Delta\mathbf{r}|)\right]
= \exp\!\left[-3.44\left(\frac{|\Delta\mathbf{r}|}{r_{0,sw}}\right)^{5/3}\right]
$$

**절차**:
1. 각 realization의 collimated output field에 circular aperture mask 적용
   (`D_\text{aperture}`가 주어지면 해당 직경, 아니면 full-window mask)
2. Masked field에 대해 2D MCF 계산
3. MCF를 앙상블 평균
4. 정규화하여 MCDOC (modulus of complex degree of coherence) 획득
5. Reference coherence factor와 1D slice 및 $e^{-1}$ width 비교
6. 결과는 "reference-consistent / partial / weak agreement"로 보고

**MCF 계산** (Listing 9.8, line 36):
$$
\Gamma(\Delta\mathbf{r}) = \langle U(\mathbf{r})\,U^*(\mathbf{r}+\Delta\mathbf{r})\rangle
$$

구현: aperture mask를 포함한 cross-correlation via FFT (`corr2_ft(U, U, mask, delta_n)` 형태).

**가우시안 빔에 대한 주의**:
이론 coherence factor는 평면파/구면파 모델에서 직접 유도된 식이므로,
가우시안 빔 결과와의 비교는 reference-level consistency check로 해석.
즉, 본 검증은 "물리적 정확성 보장"이 아니라 "수치 구현이 기대 경향을 재현하는지"를 점검하는 절차임.

---

## 6. 출력 명세 (Output Specification)

### 6.1. 파일 구조

```
output/
├── config.json              # 모든 입력/계산 파라미터 기록
├── sampling_analysis.json   # δ1, δn, N, n_scr, Δz_max, D1', D2' 등
├── screen_r0.json           # 각 screen의 r0 값, 최적화 결과
├── vacuum/
│   ├── field_vacuum.pt      # complex128 tensor [N, N]
│   └── irradiance_vacuum.pt # float64 tensor [N, N]
├── turbulence/
│   ├── field_0000.pt        # complex128 tensor [N, N]
│   ├── field_0001.pt        
│   ├── ...
│   ├── field_{n_reals-1}.pt
│   ├── irradiance_0000.pt   # float64 tensor [N, N]
│   ├── irradiance_0001.pt
│   ├── ...
│   └── irradiance_{n_reals-1}.pt
├── verification/
│   ├── structure_function.png    # Dφ vs theory plot
│   ├── coherence_factor.png      # μ vs theory plot
│   ├── verification_report.json  # pass/fail 판정 + 수치
│   └── irradiance_samples.png    # 대표 irradiance 시각화
└── coordinates.pt           # xn, yn 좌표 [N] each (float64)
```

### 6.2. 좌표 정보

관측면 좌표:
```python
xn = torch.arange(-N//2, N//2, dtype=torch.float64) * delta_n  # [m]
yn = xn.clone()
# 2D: Xn, Yn = torch.meshgrid(xn, yn, indexing='ij')
```

---

## 7. 구현 아키텍처 (Implementation Architecture)

### 7.1. 모듈 구조

```
fso_propagation/
├── __init__.py
├── config.py              # SimulationConfig dataclass
├── sampling.py            # Sampling constraint analysis (Sec 4.3)
├── atmosphere.py          # r0, σ²χ 계산 + screen r0 optimization (Sec 4.2, 4.4)
├── phase_screen.py        # FT + Subharmonic screen generation (Sec 4.5)
├── propagation.py         # Split-step angular-spectrum propagation (Sec 4.6)
├── verification.py        # Structure function + coherence factor (Sec 5)
├── ft_utils.py            # ft2, ift2, corr2_ft (PDF의 FT 컨벤션 구현)
├── main.py                # Entry point: config → sampling → propagate → verify → save
└── utils.py               # I/O, logging, plotting
```

참고: 현재 저장소에 통합할 때는 위 구조를 개념적 모듈 분해로 보고,
별도 top-level package를 만들기보다 `kim2026.*` 네임스페이스 아래로 매핑하는 것을 우선 고려.

### 7.2. 핵심 함수 시그니처

```python
# ft_utils.py
def ft2(g: torch.Tensor, delta: float) -> torch.Tensor:
    """2D FT with proper scaling. Eq: FT2(g) = fftshift(fft2(ifftshift(g))) * delta^2"""
    
def ift2(G: torch.Tensor, delta_f: float) -> torch.Tensor:
    """2D IFT with proper scaling. Eq: IFT2(G) = ifftshift(ifft2(fftshift(G))) * N^2 * delta_f^2"""

# phase_screen.py
def ft_phase_screen(r0: float, N: int, delta: float) -> torch.Tensor:
    """Listing 9.2: FFT-based phase screen (Kolmogorov PSD)"""

def ft_sh_phase_screen(r0: float, N: int, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Listing 9.3: FT + Subharmonic phase screen. Returns (phz_lo, phz_hi)"""

# propagation.py
def ang_spec_multi_prop(
    Uin: torch.Tensor,      # [N, N] complex128
    wvl: float,
    delta1: float,
    deltan: float,
    z: torch.Tensor,        # [n-1] plane locations
    t: torch.Tensor,        # [n, N, N] phase screen + absorbing boundary
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Listing 9.1: Split-step angular-spectrum propagation.
    Returns (xn, yn, Uout)"""

# atmosphere.py
def compute_atmospheric_params(k: float, Cn2: float, Dz: float) -> dict:
    """Compute r0_sw, r0_pw, sigma2_chi_sw"""

def optimize_screen_r0(
    r0_sw: float, sigma2_chi: float, k: float, Dz: float, n_scr: int
) -> np.ndarray:
    """Listing 9.5: Constrained optimization for screen r0 values"""

# sampling.py
def analyze_sampling(config: SimulationConfig) -> SamplingResult:
    """Sec 4.3: Compute delta1, N, n_scr automatically"""

# verification.py
def verify_phase_screens(
    phase_screens_by_plane: list[list[torch.Tensor]],
    r0_values: list[float],
    delta_values: list[float],
) -> dict:
    """Sec 5.1: Per-plane structure function verification"""

def verify_coherence_factor(
    fields: list[torch.Tensor],
    delta_n: float,
    aperture_mask: torch.Tensor | None = None,
    r0_ref: float | None = None,
) -> dict:
    """Sec 5.2: Masked coherence-factor reference check"""
```

### 7.3. GPU 메모리 관리

A100 40GB에서의 메모리 예산:

| N | complex128 [N,N] | float64 [N,N] | n_scr screens | 총 추정 |
|---|---|---|---|---|
| 512 | 4 MB | 2 MB | 11 × 4 MB = 44 MB | ~100 MB |
| 1024 | 16 MB | 8 MB | 11 × 16 MB = 176 MB | ~400 MB |
| 2048 | 64 MB | 32 MB | 11 × 64 MB = 704 MB | ~1.5 GB |
| 4096 | 256 MB | 128 MB | 11 × 256 MB = 2.8 GB | ~6 GB |

→ N ≤ 4096에서 A100 40GB 충분. N=8192 이상은 메모리 주의 필요.

**전략**:
- Phase screen은 매 realization마다 생성 후 즉시 전파에 사용, 저장하지 않음
- 출력 field는 생성 즉시 CPU로 이동 후 `.pt` 파일로 저장
- 전파 중간 결과는 GPU에 유지 (in-place 연산 최대 활용)

### 7.4. 수치 정밀도

- **Reference run은 complex128 (= float64 real + float64 imag)** 사용
- 이유: 이차 위상 인자 $\exp(ikr^2/2\Delta z)$에서 $kr^2$이 매우 클 수 있음
  - 예: $k = 4.05\times10^6$ rad/m, $r = 2$ m → $kr^2 \sim 1.6\times10^7$ rad
  - float32의 유효 자릿수 7자리로는 위상 정보 소실
  - float64의 유효 자릿수 15자리로 충분
- Phase screen 생성의 난수도 float64
- FFT도 complex128로 수행 (`torch.fft.fft2`는 입력 dtype 유지)
- 대량 Monte-Carlo 생성이 목표일 경우 optional fast path로 complex64를 둘 수 있으나,
  이 경우 vacuum field, structure function, coherence-factor 지표가 reference run과 일치함을 먼저 확인

---

## 8. 실행 흐름 (Execution Flow)

```
1. 입력 파라미터 로드 (SimulationConfig)
   ├─ λ = 1550 nm 고정
   ├─ θ_div → w0 역산
   └─ D1 = 4*w0 설정

2. 대기 파라미터 계산
   ├─ r0_pw, r0_sw (Eq 9.42, 9.43)
   ├─ σ²_χ,sw (Eq 9.64)
   └─ Weak fluctuation 체크 (σ²χ < 0.25?)

3. Sampling analysis (자동)
   ├─ D1', D2' 계산 (Eq 9.84-9.85, c=2)
   ├─ δ1 계산 (Constraint 1,3 교집합)
   ├─ N 계산 (Constraint 2, 2의 거듭제곱)
   ├─ Δz_max, n_scr 계산 (Constraint 4)
   └─ 모든 constraint 만족 확인

4. Screen r0 최적화 (SciPy)
   ├─ A 행렬 구성
   ├─ b 벡터 구성
   ├─ SLSQP 최적화
   └─ r0_sw, σ²χ 복원 검증

5. Vacuum 전파
   ├─ 가우시안 빔 source 생성
   ├─ ang_spec_multi_prop (T=1)
   └─ 저장: field_vacuum.pt, irradiance_vacuum.pt

6. 난류 전파 (n_reals 반복)
   ├─ for j in range(n_reals):
   │   ├─ n_scr개 phase screen 생성 (FT+SH, 각각의 r0_i, delta_i)
   │   ├─ T = sg * exp(i*phz)
   │   ├─ ang_spec_multi_prop
   │   ├─ 저장: field_{j}.pt, irradiance_{j}.pt
   │   └─ (검증용) MCF 누적
   └─ end

7. 자동 검증
   ├─ Phase screen structure function vs theory
   ├─ Coherence factor vs theory
   ├─ 결과 플롯 생성
   └─ verification_report.json 저장

8. 완료 보고
   └─ config.json, sampling_analysis.json, screen_r0.json 저장
```

---

## 9. FT 컨벤션 상세 (Critical Implementation Detail)

PDF의 FT 컨벤션을 PyTorch FFT에 정확히 매핑하는 것이 **가장 중요한 구현 세부사항**.

### 9.1. PDF의 ft2 / ift2 정의

PDF에서 사용하는 2D DFT (연속 FT의 이산 근사):

**순방향 (ft2)**:
$$
G[p,q] = \delta^2 \sum_{m=0}^{N-1}\sum_{n=0}^{N-1} g[m,n]\,\exp\!\bigl[-i2\pi(mp/N + nq/N)\bigr]
$$

**역방향 (ift2)**:
$$
g[m,n] = \Delta f^2 \sum_{p=0}^{N-1}\sum_{q=0}^{N-1} G[p,q]\,\exp\!\bigl[i2\pi(mp/N + nq/N)\bigr]
$$

여기서 $\Delta f = 1/(N\delta)$.

**PyTorch 매핑**:
```python
def ft2(g, delta):
    """PDF's ft2: scaled 2D FFT with center-shifted convention"""
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(g)
        )
    ) * delta**2

def ift2(G, delta_f):
    """PDF's ift2: scaled 2D IFFT with center-shifted convention"""
    N = G.shape[0]
    return torch.fft.ifftshift(
        torch.fft.ifft2(
            torch.fft.fftshift(G)
        )
    ) * (N * delta_f)**2
```

**검증**: `ift2(ft2(g, delta), 1/(N*delta))` ≈ `g` (수치 오차 < 1e-10)

### 9.2. Listing 9.1의 FT 사용 패턴

Listing 9.1, lines 37-39:
```matlab
Uin = sg .* t(:,:,idx+1) ...
    .* ift2(Q2 .* ft2(Uin / m(idx), delta(idx)), deltaf);
```

PyTorch 직역:
```python
G = ft2(U / m[i], delta[i])          # 순방향 FT
G = Q2 * G                           # 주파수 영역 곱셈
U_prop = ift2(G, deltaf)             # 역방향 FT
U = sg * T[i+1] * U_prop             # absorbing boundary + phase screen
```

---

## 10. 검증 체크리스트 (Verification Checklist)

구현 완료 후 반드시 통과해야 하는 테스트:

### 10.1. 단위 테스트

| # | 테스트 | 판정 기준 |
|---|--------|-----------|
| 1 | ft2/ift2 왕복 | max error < 1e-10 |
| 2 | ft2(delta function) = constant | max relative error < 1e-10 |
| 3 | Parseval's theorem: ∫|g|²dx = ∫|G|²df | relative error < 1e-10 |
| 4 | Phase screen mean ≈ 0 | |mean| < 0.1 rad |
| 5 | Gaussian beam vacuum propagation vs analytic | irradiance relative error < 5% in ROI |

### 10.2. 물리 검증

| # | 테스트 | 판정 기준 |
|---|--------|-----------|
| 6 | Plane-wise phase screen Dφ(r) vs 6.88(r/r0,i)^{5/3} | each valid plane: avg relative error < 15% |
| 7 | Masked coherence factor e⁻¹ width vs ρ0(ref) | reference-consistent: within 20% |
| 8 | Vacuum irradiance: Gaussian profile 유지 | qualitative check |
| 9 | Turbulent irradiance: speckle pattern 확인 | qualitative check |
| 10 | σ²χ < 0.25 → weak fluctuation 조건 확인 | informational |

### 10.3. 수치 안정성

| # | 테스트 | 판정 기준 |
|---|--------|-----------|
| 11 | 전파 후 총 에너지 보존 (vacuum) | < 5% 손실 (absorbing boundary 고려) |
| 12 | Phase screen r0 최적화 수렴 | residual < 1e-6 |
| 13 | N, δ1 선택이 모든 constraint 만족 | Boolean pass |

---

## 11. 향후 확장 (Future Extensions)

현재 PRD 범위 밖이지만, 아키텍처에서 고려할 확장:

1. **Modified von Kármán PSD**: `phase_screen.py`에 l0, L0 파라미터 추가
2. **Non-constant Cn² profile**: `atmosphere.py`에 Cn2(z) 함수 입력 지원
3. **Temporal evolution**: Taylor frozen-turbulence (Eq 9.19) 기반 screen 이동
4. **Adaptive optics**: 수신단 wavefront sensing + correction 루프
5. **Batch GPU 처리**: 여러 realization을 batch로 동시 전파 (GPU 메모리 허용 시)

---

## 부록 A. 수식 교차 참조표

| 이 PRD의 수식 | PDF 원본 | 설명 |
|---|---|---|
| §4.2.1 r0_pw | Eq (9.42) | 평면파 Fried parameter |
| §4.2.1 r0_sw | Eq (9.43) | 구면파 Fried parameter |
| §4.2.2 σ²χ,sw | Eq (9.64) | 구면파 log-amplitude variance |
| §4.3.1 D1', D2' | Eq (9.84), (9.85) | 난류 beam spreading 보정 |
| §4.3.2 Constraint 1 | Eq (9.86) | δn 상한 |
| §4.3.3 Constraint 2 | Eq (9.87) | N 하한 |
| §4.3.2 Constraint 3 | Eq (9.88) | δ2 범위 (= Eq 8.22) |
| §4.3.4 Constraint 4 | Eq (8.24), (9.89), (9.90) | Δzmax, nmin |
| §4.4 r0 분배 | Eq (9.72), (9.74), (9.75) | Screen r0 최적화 |
| §4.5.1 Phase PSD | Eq (9.52) | Kolmogorov phase PSD (f domain) |
| §4.5.2 Subharmonic | Eq (9.81) | 저주파 보정 |
| §4.6 전파 | Eq (9.3) = Eq (8.18)+T | Split-step 알고리즘 |
| §5.1 Dφ(r) | Eq (9.44) | Kolmogorov structure function |
| §5.2 μ(Δr) | Eq (9.32) + Eq (9.44) | Collimated-output reference coherence factor |

## 부록 B. 가우시안 빔 w0 역산

Full-angle far-field divergence $\theta_\text{div}$ (far-field half-angle = $\theta_\text{div}/2$):

$$
\theta_\text{half} = \frac{\lambda}{\pi w_0}
\quad\Rightarrow\quad
w_0 = \frac{\lambda}{\pi \cdot \theta_\text{div}/2} = \frac{2\lambda}{\pi\,\theta_\text{div}}
$$

Rayleigh range: $z_R = \pi w_0^2 / \lambda$.

콜리메이티드 가우시안 빔의 source-plane field:
$$
U(\mathbf{r}_1) = A_0 \exp\!\left(-\frac{|\mathbf{r}_1|^2}{w_0^2}\right)
$$

여기서 $A_0$는 진폭 정규화 상수. 총 파워 $P$에 대해:
$$
A_0 = \sqrt{\frac{2P}{\pi w_0^2}}
$$

본 시뮬레이터에서는 상대 비교가 목적이므로 $A_0 = 1$ 사용 가능.

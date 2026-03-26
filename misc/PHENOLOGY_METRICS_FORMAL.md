# Phenology Metrics and Deciduousness Scoring (Formal Definitions)

This document formalizes the phenology-related signals computed by the pipeline in `src/notebooks/Phenology_signals_10Mar26.ipynb`, matching the implementation (masks, thresholds, normalization constants, and edge cases).

## 1) Notation and data model

- Trees (consensus crowns): $t\in\mathcal{T}$
- Orthomosaics (OM IDs): $\mathcal{O}=\{o_1,\dots,o_T\}$ in increasing order
- For each $(t,o)$, an RGB crown patch is extracted: $P_{t,o}\in\mathbb{R}^{H\times W\times 3}$.

Pixels are indexed by $p\in\{1,\dots,HW\}$. Let $(R_p,G_p,B_p)$ denote raw RGB values (expected in $[0,255]$).
The implementation clips and rescales to $[0,1]$:

$$
(r_p,g_p,b_p)=\frac{1}{255}\,\bigl(\mathrm{clip}(R_p,0,255),\,\mathrm{clip}(G_p,0,255),\,\mathrm{clip}(B_p,0,255)\bigr).
$$

## 2) Pixel validity and counts

A pixel is **valid** if all channels are finite and the raw-channel sum is positive:

$$
\mathbb{1}_{\mathrm{finite}}(p)=\mathbb{1}[R_p,G_p,B_p\ \text{finite}],\quad
\mathbb{1}_{\mathrm{nonzero}}(p)=\mathbb{1}[R_p+G_p+B_p>0],\quad
\mathbb{1}_{\mathrm{valid}}(p)=\mathbb{1}_{\mathrm{finite}}(p)\,\mathbb{1}_{\mathrm{nonzero}}(p).
$$

Define:

$$
N = HW,\qquad N_{\mathrm{valid}}=\sum_{p=1}^{N} \mathbb{1}_{\mathrm{valid}}(p),\qquad
\mathrm{valid\_pixel\_fraction}(t,o)=\frac{N_{\mathrm{valid}}}{N}.
$$

**Early exit:** if $N=0$ or $N_{\mathrm{valid}}<20$, extraction returns only `valid_pixel_fraction`.

## 3) Chromatic signals (GCC, RCC, ExG)

Let $S_p=r_p+g_p+b_p$. Safe division uses $\varepsilon=10^{-8}$:

$$\mathrm{safe\_div}(a,b)=\frac{a}{b+\varepsilon}.$$

Per-pixel:

$$
\mathrm{GCC}_p=\frac{g_p}{S_p+\varepsilon},\quad
\mathrm{RCC}_p=\frac{r_p}{S_p+\varepsilon},\quad
\mathrm{ExG}_p=2g_p-r_p-b_p.
$$

Patch-level means are computed over valid pixels:

$$
\mathrm{gcc\_mean}(t,o)=\frac{1}{N_{\mathrm{valid}}}\sum_{p=1}^{N}\mathbb{1}_{\mathrm{valid}}(p)\,\mathrm{GCC}_p,
\quad
\mathrm{rcc\_mean}(t,o)=\frac{1}{N_{\mathrm{valid}}}\sum_{p=1}^{N}\mathbb{1}_{\mathrm{valid}}(p)\,\mathrm{RCC}_p,
$$

$$
\mathrm{exg\_mean}(t,o)=\frac{1}{N_{\mathrm{valid}}}\sum_{p=1}^{N}\mathbb{1}_{\mathrm{valid}}(p)\,\mathrm{ExG}_p.
$$

## 4) HSV vegetation and shadow fractions

HSV $(h_p,s_p,v_p)\in[0,1]^3$ is computed from $(r_p,g_p,b_p)$ using `matplotlib.colors.rgb_to_hsv`.

**Shadow/dark mask:**

$$\mathbb{1}_{\mathrm{dark}}(p)=\mathbb{1}[v_p<0.12].$$

Shadow fraction over valid pixels:

$$
\mathrm{shadow\_fraction}(t,o)=\frac{1}{N_{\mathrm{valid}}}\sum_{p=1}^{N}\mathbb{1}_{\mathrm{valid}}(p)\,\mathbb{1}_{\mathrm{dark}}(p).
$$

**Vegetation-like mask:**

$$
\mathbb{1}_{\mathrm{veg}}(p)=\mathbb{1}_{\mathrm{valid}}(p)\,\mathbb{1}[0.18\le h_p\le 0.48] \,\mathbb{1}[s_p\ge 0.15] \,\mathbb{1}[v_p\ge 0.12].
$$

Vegetation fraction:

$$
\mathrm{veg\_fraction\_hsv}(t,o)=\frac{1}{N_{\mathrm{valid}}}\sum_{p=1}^{N}\mathbb{1}_{\mathrm{veg}}(p).
$$

## 5) Channel statistics and coefficients of variation

Let $\mathcal{V}=\{p: \mathbb{1}_{\mathrm{valid}}(p)=1\}$.

Means:

$$
\mathrm{r\_mean}=\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}} r_p,\quad
\mathrm{g\_mean}=\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}} g_p,\quad
\mathrm{b\_mean}=\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}} b_p.
$$

Population std (NumPy default `ddof=0`):

$$
\mathrm{r\_std}=\sqrt{\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}}(r_p-\mathrm{r\_mean})^2},\ \text{etc.}
$$

CVs (safe-divided):

$$
\mathrm{r\_cv}=\frac{\mathrm{r\_std}}{\mathrm{r\_mean}+\varepsilon},\quad
\mathrm{g\_cv}=\frac{\mathrm{g\_std}}{\mathrm{g\_mean}+\varepsilon},\quad
\mathrm{b\_cv}=\frac{\mathrm{b\_std}}{\mathrm{b\_mean}+\varepsilon}.
$$

## 6) Grayscale texture metrics

Grayscale luminance:

$$\mathrm{gray}_p = 0.299\,r_p + 0.587\,g_p + 0.114\,b_p.$$

Gaussian smoothing:

$$\mathrm{gray\_smooth}=\mathrm{GaussianBlur}(\mathrm{gray};\ \text{kernel }(5,5)).$$

Std on valid pixels:

$$
\mathrm{gray\_std}=\mathrm{StdDev}(\{\mathrm{gray}_p: p\in\mathcal{V}\}),\quad
\mathrm{gray\_std\_smooth}=\mathrm{StdDev}(\{\mathrm{gray\_smooth}_p: p\in\mathcal{V}\}).
$$

### 6.1) Gray entropy (two versions in notebook)

Histogram over $[0,1]$ with $B=64$ bins.

- Earlier version (later overwritten): `density=True`, then $H=-\sum_b h_b\log_2 h_b$ for bins with $h_b>0$.
- Final intended version (entropy-fix cell): counts histogram (`density=False`) → probabilities $p_b=c_b/\sum c_b$.

Shannon entropy:

$$\mathrm{gray\_entropy}=-\sum_{b=1}^{B} p_b\log_2 p_b\quad (p_b>0).$$

Note: exported feature CSVs reflect whichever definition was active when extraction was run.

### 6.2) Laplacian variance

Convert to uint8: $\mathrm{gray\_u8}=\mathrm{uint8}(\mathrm{clip}(255\,\mathrm{gray},0,255))$.

Then:

$$\mathrm{laplacian\_var}=\mathrm{Var}(\Delta\,\mathrm{gray\_u8}),$$

where $\Delta$ is `cv2.Laplacian(gray_u8, CV_64F)` and variance is over all pixels.

## 7) GLCM texture metrics (optional)

If scikit-image is available, quantize to $L=32$ levels:

$$q_p=\mathrm{clip}(\lfloor 31\,\mathrm{gray}_p\rfloor,0,31)\in\{0,\dots,31\}.$$

Compute GLCMs with distance $d=1$ and angles $\Theta=\{0,\pi/4,\pi/2,3\pi/4\}$ using `symmetric=True`, `normed=True`.

Reported metrics are the mean over angles:

$$
\mathrm{glcm\_contrast}=\frac{1}{|\Theta|}\sum_{\theta\in\Theta}\phi_{\mathrm{contrast}}(G^{(\theta)}),\quad
\mathrm{glcm\_homogeneity}=\frac{1}{|\Theta|}\sum_{\theta\in\Theta}\phi_{\mathrm{homogeneity}}(G^{(\theta)}),\quad
\mathrm{glcm\_energy}=\frac{1}{|\Theta|}\sum_{\theta\in\Theta}\phi_{\mathrm{energy}}(G^{(\theta)}).
$$

If unavailable or failing, these are set to NaN.

## 8) Observation-level QC (`is_bad_observation`)

A $(t,o)$ observation is flagged bad iff:

$$
\mathrm{valid\_pixel\_fraction}<0.60\ \lor\ 
\mathrm{shadow\_fraction}>0.55\ \lor\ 
\mathrm{laplacian\_var}<25.
$$

## 9) Date-wise robust normalization

For any feature $x_{t,o}$, for each OM $o$:

$$\mathrm{med}_o=\mathrm{median}_t\,x_{t,o},\quad \mathrm{MAD}_o=\mathrm{median}_t\,|x_{t,o}-\mathrm{med}_o|.$$

Robust z-score:

$$\mathrm{rz\_date}_{t,o}=\frac{x_{t,o}-\mathrm{med}_o}{1.4826\,\mathrm{MAD}_o+10^{-8}}.$$

Also computed: per-OM percentile rank `pct_date` via groupwise `rank(pct=True)`.

## 10) Per-tree temporal descriptors (amplitude and slope)

Computed on **clean** observations only: $\mathcal{O}^{\mathrm{clean}}_t=\{o: \mathrm{is\_bad\_observation}(t,o)=0\}$.

Amplitude:

$$\mathrm{amplitude}_t(x)=\max_{o\in\mathcal{O}^{\mathrm{clean}}_t}x_{t,o}-\min_{o\in\mathcal{O}^{\mathrm{clean}}_t}x_{t,o}.$$

Slope per OM (polyfit on $(o, x_{t,o})$): record the fitted linear coefficient.

## 11) Time-series interpolation + per-tree min–max normalization

### 11.1) Pivot and NaN-out bad observations

For each metric, build a $\text{tree}\times\text{OM}$ table and set bad observations to NaN.

### 11.2) Linear interpolation with edge fill

For a tree’s series $y_k$ at OM IDs $\xi_k=o_k$:

- If there are $
\ge 2$ finite points: linear interpolation across gaps, and **edge-fill** outside the observed range using the first/last finite values.
- If exactly 1 finite point: fill all missing with that value.
- If 0 finite points: fill all entries with 0.

### 11.3) Per-tree min–max normalization

For each tree and metric (after interpolation):

$$
\bar{x}_t[k]=\begin{cases}
\dfrac{\hat{x}_t[k]-\min_{k}\hat{x}_t[k]}{\max_{k}\hat{x}_t[k]-\min_{k}\hat{x}_t[k]} & \text{if range}>10^{-9}\\
0 & \text{otherwise.}
\end{cases}
$$

## 12) Phenometrics used for post-hoc labelling

From each interpolated series $\hat{x}_t$:

- amplitude: $\max-\min$
- peak OM: $o_{k^{\mathrm{pk}}}$ where $k^{\mathrm{pk}}=\arg\max_{k}\,\hat{x}_t[k]$
- trough OM: $o_{k^{\mathrm{tr}}}$ where $k^{\mathrm{tr}}=\arg\min_{k}\,\hat{x}_t[k]$
- slope: linear fit vs **time index** $k\in\{0,\dots,T-1\}$ (not OM ID)

Directional changes are computed on the normalized series:

- $d_k=\bar{x}[k+1]-\bar{x}[k]$
- keep only steps with $|d_k|>0.05$
- `dir_changes` = number of sign flips in the remaining sequence.

## 13) Leaf-shed classifier

Constants:

- `VEG_MIN_THRESH = 0.45`
- `SCORE_THRESH = 0.70`
- phenophase thresholds on normalized veg: `LEAFON_NORM=0.65`, `LEAFOFF_NORM=0.35`

Signals:

- primary: `veg_fraction_hsv`
- secondary: `gcc_mean`
- optional texture: `laplacian_var`

### 13.1) 90th-percentile amplitude normalizers

Across trees:

- $A_{90}^{\mathrm{veg}} = \mathrm{percentile}_{90}(\max\hat{v}-\min\hat{v})$
- $A_{90}^{\mathrm{gcc}} = \mathrm{percentile}_{90}(\max\hat{g}-\min\hat{g})$
- texture analogous if used

(fallback to 1.0 if percentile evaluates to 0)

### 13.2) Score components

Within tree loop, veg is clipped: $\hat{v}[k]=\mathrm{clip}(\hat{x}^{\mathrm{veg}}[k],0,1)$.

- $s_{\mathrm{veg\_amp}}=\min(1,\mathrm{veg\_amp}/A_{90}^{\mathrm{veg}})$
- $s_{\mathrm{gcc\_amp}}=\min(1,\mathrm{gcc\_amp}/A_{90}^{\mathrm{gcc}})$
- depth component:

$$
 s_{\mathrm{depth}}=\begin{cases}
\dfrac{0.45-\mathrm{veg\_min}}{0.45} & \mathrm{veg\_min}<0.45\\
0 & \text{otherwise}
\end{cases}
$$

- $s_{\mathrm{tex}}=\min(1,\mathrm{tex\_amp}/A_{90}^{\mathrm{tex}})$ if texture present, else 0

### 13.3) Deciduousness score and decision

If texture present:

$$\mathrm{DS}=0.35s_{\mathrm{veg\_amp}}+0.30s_{\mathrm{depth}}+0.25s_{\mathrm{gcc\_amp}}+0.10s_{\mathrm{tex}}.$$

Else:

$$\mathrm{DS}=0.40s_{\mathrm{veg\_amp}}+0.35s_{\mathrm{depth}}+0.25s_{\mathrm{gcc\_amp}}.$$

Classification:

$$\mathrm{is\_deciduous}=\mathbb{1}[\mathrm{DS}\ge 0.70].$$

### 13.4) Phenophase states per OM

Let $v_n[k]$ be the per-tree min–max normalized veg series.

- If not deciduous: state = `stable` for all OMs
- Else:
  - if $v_n\ge 0.65$: `leaf_on`
  - if $v_n\le 0.35$: `leaf_off`
  - else: `transitioning`

### 13.5) Event timing

Let $k_{\min}=\arg\min\hat{v}[k]$ and $k_{\max}=\arg\max\hat{v}[k]$.

- `full_leaf_off_om = o_{kOff}`
- `leaf_on_peak_om = o_{kOn}`

If deciduous:

- `leaf_off_start_om`: scan backward from $k_{\min}$ to find the last `leaf_on` at index $k$; return $o_{\min(k+1,T)}$.
- `leaf_on_return_om`: scan forward from $k_{\min}$ to find first `leaf_on` at index $k$; return $o_k$.

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.fftpack import dct, idct

def get_luma_block(image_path, block_size=16):
    # 이미지가 없을 경우 테스트용 합성 데이터 생성
    if not os.path.exists(image_path):
        print(f"[{image_path}] 파일을 찾을 수 없어 합성 데이터를 생성합니다.")
        x = np.linspace(0, 1, block_size)
        y = np.linspace(0, 1, block_size)
        X, Y = np.meshgrid(x, y)
        structure = 100 * np.sin(X * 5) + 50 * np.cos(Y * 3) + 128
        noise = np.random.normal(0, 10, (block_size, block_size))
        block = np.clip(structure + noise, 0, 255)
        return block
    
    # 이미지 로드 및 루마(L) 변환
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    h, w = img_array.shape
    # 중앙 부분에서 16x16 블록 추출
    sy, sx = h//2 - block_size//2, w//2 - block_size//2
    return img_array[sy:sy+block_size, sx:sx+block_size].astype(np.float64)

# 방법 1: DCT (주파수 분리) - 압축 표준에서 주로 사용
def dct_decompose(block, cutoff=4):
    # 2D DCT 수행
    c = dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    # 저주파(구조) 성분만 추출 (cutoff x cutoff 영역)
    c_low = np.zeros_like(c)
    c_low[:cutoff, :cutoff] = c[:cutoff, :cutoff]
    
    structure = idct(idct(c_low.T, norm='ortho').T, norm='ortho')
    noise = block - structure
    return structure, noise

# 방법 2: 가우시안 필터 (공간적 평활화)
def filter_decompose(block, sigma=1.5):
    structure = gaussian_filter(block, sigma=sigma)
    noise = block - structure
    return structure, noise

# 에너지 계산 함수
def get_energy(data):
    return np.sum(data**2)

# 실행 및 시각화
block = get_luma_block('1.png')

# 분해 수행
s_dct, n_dct = dct_decompose(block, cutoff=4)
s_fil, n_fil = filter_decompose(block, sigma=1.5)

# 시각화 설정
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
methods = [('DCT (Frequency)', s_dct, n_dct), ('Gaussian (Spatial)', s_fil, n_fil)]

for i, (name, s, n) in enumerate(methods):
    e_s, e_n = get_energy(s), get_energy(n)
    
    axes[i, 0].imshow(block, cmap='gray')
    axes[i, 0].set_title("Original Block")
    
    axes[i, 1].imshow(s, cmap='gray')
    axes[i, 1].set_title(f"{name} Structure\nEnergy: {e_s:.1e}")
    
    # 노이즈는 시각적 확인을 위해 오프셋 조정
    axes[i, 2].imshow(n, cmap='gray', vmin=-20, vmax=20)
    axes[i, 2].set_title(f"{name} Noise\nEnergy: {e_n:.1e}")

plt.tight_layout()
plt.show()

print(f"[DCT] 구조 에너지: {get_energy(s_dct):.2f}, 노이즈 에너지: {get_energy(n_dct):.2f}")
print(f"[Filter] 구조 에너지: {get_energy(s_fil):.2f}, 노이즈 에너지: {get_energy(n_fil):.2f}")













# -----------------------------
# DCT / IDCT helpers
# -----------------------------
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(coeff):
    return idct(idct(coeff.T, norm='ortho').T, norm='ortho')


# -----------------------------
# Noise sigma estimate from high-frequency DCT coefficients
# -----------------------------
def estimate_noise_sigma_dct(block, hf_size=6):
    """
    block: 2D grayscale block
    hf_size: use bottom-right hf_size x hf_size DCT region for robust noise estimate
    
    Returns:
        sigma_n: estimated noise std in pixel domain (approximate)
    """
    c = dct2(block)

    h, w = c.shape
    hf = c[h-hf_size:h, w-hf_size:w].copy().reshape(-1)

    # Remove DC-like accidental contamination if any
    hf = hf[np.isfinite(hf)]
    if hf.size == 0:
        return 1.0

    # Robust MAD estimate
    med = np.median(hf)
    mad = np.median(np.abs(hf - med))
    sigma_n = mad / 0.6745 if mad > 0 else 1.0

    # avoid zero
    sigma_n = max(float(sigma_n), 1e-3)
    return sigma_n


# -----------------------------
# Method 3: DCT adaptive shrinkage
# -----------------------------
def dct_adaptive_shrinkage_decompose(
    block,
    method='wiener',          # 'wiener' or 'soft'
    sigma_n=None,             # if None, estimated automatically
    soft_k=2.7,               # threshold multiplier for soft-threshold
    preserve_dc=True,
    freq_weight=True
):
    """
    block: 2D grayscale block (float64 recommended)

    Returns:
        structure: reconstructed low-noise / structural component
        noise: residual = block - structure
        info: dict containing sigma_n and DCT coeffs
    """
    c = dct2(block)

    if sigma_n is None:
        sigma_n = estimate_noise_sigma_dct(block)

    # Frequency weighting:
    # higher frequency -> slightly more aggressive shrinkage
    h, w = c.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    rr = np.sqrt(uu**2 + vv**2)

    if freq_weight:
        # normalized frequency weight in [1, ~2]
        rr_norm = rr / (np.max(rr) + 1e-8)
        weight = 1.0 + rr_norm
    else:
        weight = np.ones_like(c)

    c_hat = np.zeros_like(c)

    if method == 'soft':
        # soft threshold per coefficient
        T = soft_k * sigma_n * weight
        c_hat = np.sign(c) * np.maximum(np.abs(c) - T, 0.0)

    elif method == 'wiener':
        # Wiener-like shrinkage:
        # gain = signal_var / (signal_var + noise_var)
        # Here we approximate local signal power with coeff power itself.
        noise_var = (sigma_n ** 2) * (weight ** 2)
        gain = (c ** 2) / (c ** 2 + noise_var + 1e-12)
        c_hat = gain * c

    else:
        raise ValueError("method must be 'wiener' or 'soft'")

    if preserve_dc:
        c_hat[0, 0] = c[0, 0]

    structure = idct2(c_hat)
    noise = block - structure

    info = {
        'sigma_n': sigma_n,
        'coeff': c,
        'coeff_shrunk': c_hat
    }
    return structure, noise, info






# 방법 3: DCT adaptive shrinkage
s_ads_w, n_ads_w, info_w = dct_adaptive_shrinkage_decompose(
    block,
    method='wiener',
    sigma_n=None,      # 자동 추정
    preserve_dc=True,
    freq_weight=True
)

s_ads_s, n_ads_s, info_s = dct_adaptive_shrinkage_decompose(
    block,
    method='soft',
    sigma_n=None,      # 자동 추정
    soft_k=2.7,
    preserve_dc=True,
    freq_weight=True
)



# 시각화 설정
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
methods = [
    ('DCT Hard Cutoff', s_dct, n_dct, None),
    ('Gaussian', s_fil, n_fil, None),
    ('DCT Adaptive Wiener', s_ads_w, n_ads_w, info_w),
    ('DCT Adaptive Soft', s_ads_s, n_ads_s, info_s),
]

for i, (name, s, n, info) in enumerate(methods):
    e_s, e_n = get_energy(s), get_energy(n)

    axes[i, 0].imshow(block, cmap='gray', vmin=0, vmax=255)
    axes[i, 0].set_title("Original Block")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(s, cmap='gray', vmin=0, vmax=255)
    if info is not None:
        axes[i, 1].set_title(
            f"{name} Structure\nEnergy: {e_s:.1e}, sigma={info['sigma_n']:.2f}"
        )
    else:
        axes[i, 1].set_title(f"{name} Structure\nEnergy: {e_s:.1e}")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(n, cmap='gray', vmin=-20, vmax=20)
    axes[i, 2].set_title(f"{name} Residual\nEnergy: {e_n:.1e}")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()

print(f"[DCT Hard] structure: {get_energy(s_dct):.2f}, residual: {get_energy(n_dct):.2f}")
print(f"[Gaussian] structure: {get_energy(s_fil):.2f}, residual: {get_energy(n_fil):.2f}")
print(f"[Adaptive Wiener] structure: {get_energy(s_ads_w):.2f}, residual: {get_energy(n_ads_w):.2f}, sigma={info_w['sigma_n']:.3f}")
print(f"[Adaptive Soft] structure: {get_energy(s_ads_s):.2f}, residual: {get_energy(n_ads_s):.2f}, sigma={info_s['sigma_n']:.3f}")

















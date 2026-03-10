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

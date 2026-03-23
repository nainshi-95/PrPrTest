import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================
# 1. CSV 읽기
# =========================
csv_path = "your_file.csv"   # <- 여기 경로 수정

df = pd.read_csv(csv_path)

if df.shape[1] != 3:
    raise ValueError(f"CSV에는 정확히 3개의 column이 있어야 합니다. 현재: {df.shape[1]}개")

# column 이름 자동 사용
x_name, y_name, z_name = df.columns

x = df[x_name].to_numpy()
y = df[y_name].to_numpy()
z = df[z_name].to_numpy()

# =========================
# 2. 다중선형회귀로 추세면 적합
#    z = a*x + b*y + c
# =========================
X = np.column_stack((x, y))
model = LinearRegression()
model.fit(X, z)

z_pred = model.predict(X)
r2 = model.score(X, z)

a, b = model.coef_
c = model.intercept_

print("=== Regression Result ===")
print(f"Model: {z_name} = {a:.6f}*{x_name} + {b:.6f}*{y_name} + {c:.6f}")
print(f"R^2 = {r2:.6f}")

# =========================
# 3. 추세면 그리기 위한 grid 생성
# =========================
x_range = np.linspace(x.min(), x.max(), 30)
y_range = np.linspace(y.min(), y.max(), 30)
xx, yy = np.meshgrid(x_range, y_range)
zz = model.predict(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

# =========================
# 4. 3D scatter + 추세면 plot
# =========================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# scatter
ax.scatter(x, y, z, s=40, alpha=0.8, label="Data")

# trend plane
ax.plot_surface(xx, yy, zz, alpha=0.4)

ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel(z_name)
ax.set_title(f"3D Scatter with Trend Plane\nR^2 = {r2:.6f}")

plt.tight_layout()
plt.show()










import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# 1. CSV 읽기
# =========================
csv_path = "your_file.csv"   # 경로 수정
df = pd.read_csv(csv_path)

if df.shape[1] != 3:
    raise ValueError(f"CSV에는 정확히 3개의 column이 있어야 합니다. 현재: {df.shape[1]}개")

x_name, y_name, z_name = df.columns

x = df[x_name].to_numpy()
y = df[y_name].to_numpy()
z = df[z_name].to_numpy()

X = np.column_stack((x, y))

# =========================
# 2. 2차 polynomial fitting
#    z = ax^2 + by^2 + cxy + dx + ey + f
# =========================
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, z)

z_pred = model.predict(X_poly)
r2 = r2_score(z, z_pred)

feature_names = poly.get_feature_names_out([x_name, y_name])
coef = model.coef_
intercept = model.intercept_

print("=== 2nd Polynomial Regression Result ===")
terms = [f"{intercept:.6f}"]
for name, c in zip(feature_names[1:], coef[1:]):  # bias 제외
    terms.append(f"({c:.6f})*{name}")
print(f"{z_name} = " + " + ".join(terms))
print(f"R^2 = {r2:.6f}")

# =========================
# 3. surface grid 생성
# =========================
x_range = np.linspace(x.min(), x.max(), 50)
y_range = np.linspace(y.min(), y.max(), 50)
xx, yy = np.meshgrid(x_range, y_range)

grid = np.column_stack((xx.ravel(), yy.ravel()))
grid_poly = poly.transform(grid)
zz = model.predict(grid_poly).reshape(xx.shape)

# =========================
# 4. Plot
# =========================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, s=40, alpha=0.8, label="Data")
ax.plot_surface(xx, yy, zz, alpha=0.4)

ax.set_xlabel(x_name)
ax.set_ylabel(y_name)
ax.set_zlabel(z_name)
ax.set_title(f"3D Scatter with 2nd Polynomial Surface\nR^2 = {r2:.6f}")

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.font_manager as fm

# ========== 页面配置 ==========
st.set_page_config(page_title="弹簧振子的位移随时间变化规律", layout="centered")
st.title("🔔 弹簧振子的位移随时间变化规律 (最小二乘法拟合)")

st.markdown(
    """
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; font-size:16px;">
    本实验使用最小二乘法拟合弹簧振子位移随时间变化的正弦函数规律。<br>
    <b>使用说明：</b> 点击表格输入数据，可添加行或修改数值；点击下方按钮进行绘图或查看公式。
    </div>
    """,
    unsafe_allow_html=True
)

# ========== 字体设置 ==========
try:
    myfont = fm.FontProperties(fname="NotoSansSC-VariableFont_wght.ttf")
except:
    myfont = fm.FontProperties(fname="NotoSansSC-VariableFont_wght.ttf")

plt.rcParams['axes.unicode_minus'] = False

# ========== 默认数据（来自你照片里的表格） ==========
default_time = np.array([0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.30,1.40])
default_disp = np.array([-4.78,-2.40,-1.20,1.10,4.10,5.20,3.80,0.60,-2.65,-4.90,-4.70,-2.28,-0.70,2.15,3.75])

default_data = pd.DataFrame({
    "时间 (s)": default_time,
    "位移 (cm)": default_disp
})

# ========== 输入表格 ==========
data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
t_data = np.array(data["时间 (s)"])
y_data = np.array(data["位移 (cm)"])

# ========== 最小二乘法拟合函数 ==========
def fit_sine_least_squares(t, y):
    # 估计频率：用 FFT 找主频
    n = len(t)
    dt = np.mean(np.diff(t))
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_magnitude = np.abs(np.fft.rfft(y - np.mean(y)))
    freq_guess = freqs[np.argmax(fft_magnitude[1:]) + 1]  # 主频
    omega_guess = 2 * np.pi * freq_guess

    # 线性最小二乘：拟合 a*sin(ωt)+b*cos(ωt)+C
    def linear_fit(omega):
        X = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coeffs

    coeffs = linear_fit(omega_guess)
    a, b, C = coeffs
    A = np.sqrt(a**2 + b**2)
    phi = np.arctan2(b, a)
    omega = omega_guess

    # 计算 R²
    y_fit = A * np.sin(omega * t + phi) + C
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot

    return A, omega, phi, C, r2

# ========== 按钮区 ==========
col1, col2 = st.columns(2)

with col1:
    draw = st.button("🎨 绘制图像")

with col2:
    show_func = st.button("🧮 显示公式")

# ========== 绘制图像 ==========
if draw:
    A, omega, phi, C, r2 = fit_sine_least_squares(t_data, y_data)
    t_fit = np.linspace(min(t_data), max(t_data), 500)
    y_fit = A * np.sin(omega * t_fit + phi) + C

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.scatter(t_data, y_data, color="blue", label="实验数据")
    ax.plot(t_fit, y_fit, color="red", label=f"拟合曲线 (R²={r2:.3f})")
    ax.set_xlabel("时间 (s)", fontproperties=myfont)
    ax.set_ylabel("位移 (cm)", fontproperties=myfont)
    ax.legend(prop=myfont)
    ax.set_title("弹簧振子的位移随时间变化规律 (最小二乘法)", fontproperties=myfont)
    st.pyplot(fig)

# ========== 显示函数表达式 ==========
if show_func:
    A, omega, phi, C, r2 = fit_sine_least_squares(t_data, y_data)
    expr = f"x(t) = {A:.2f} · sin({omega:.2f}·t + {phi:.2f}) + {C:.2f}"
    st.markdown(
        f"<div style='text-align:center; font-size:18px; color:#444;'>"
        f"拟合函数表达式：<br><b>{expr}</b><br>"
        f"R² = {r2:.4f} （单位: cm）</div>",
        unsafe_allow_html=True
    )

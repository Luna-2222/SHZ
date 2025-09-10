import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st
import matplotlib.font_manager as fm

# 指定微软雅黑字体
myfont = fm.FontProperties(fname="NotoSansSC-VariableFont_wght.ttf")

# ================= 页面配置 =================
st.set_page_config(page_title="三角函数拟合", layout="centered")
st.title("📈 三角函数拟合工具")

st.markdown("在下方表格中输入 **时间 (s)** 和 **位移 (m)** 数据，点击按钮进行拟合。")

# ================= Matplotlib 中文字体设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先使用黑体或微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ================= 输入表格 =================
default_data = pd.DataFrame({
    "时间 (s)": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "位移 (m)": [0.0, 0.15, 0.28, 0.30, 0.25, 0.10, -0.05, -0.18, -0.28]
})

data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

t_data = np.array(data["时间 (s)"])
y_data = np.array(data["位移 (m)"])

# ================= 拟合函数 =================
def func(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

def fit_data(t, y):
    try:
        popt, _ = curve_fit(func, t, y, p0=[(max(y)-min(y))/2, 2*np.pi, 0, np.mean(y)])
        return popt
    except Exception as e:
        st.error(f"拟合失败: {e}")
        return None

# ================= 按钮交互 =================
col1, col2 = st.columns(2)

with col1:
    if st.button("🎨 绘制图像"):
        params = fit_data(t_data, y_data)
        if params is not None:
            A, omega, phi, C = params
            t_fit = np.linspace(min(t_data), max(t_data), 500)
            y_fit = func(t_fit, *params)

            fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
            ax.scatter(t_data, y_data, color="red", label="实验数据")
            ax.plot(t_fit, y_fit, color="blue", label="拟合曲线")
            ax.set_xlabel("时间 (s)",fontproperties=myfont)
            ax.set_ylabel("位移 (m)",fontproperties=myfont)
            ax.legend(prop=myfont)
            ax.set_title("三角函数拟合结果",,fontproperties=myfont)
            st.pyplot(fig)

with col2:
    if st.button("🧮 显示函数表达式"):
        params = fit_data(t_data, y_data)
        if params is not None:
            A, omega, phi, C = params
            st.success(f"拟合函数:\n\n y(t) = {A:.3f} * sin({omega:.3f} * t + {phi:.3f}) + {C:.3f}")

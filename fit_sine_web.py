import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st
import matplotlib.font_manager as fm

# 指定中文字体（你可以换成 "fonts/msyh.ttf"）
myfont = fm.FontProperties(fname="NotoSansSC-VariableFont_wght.ttf")

# ================= 页面配置 =================
st.set_page_config(page_title="三角函数拟合", layout="centered")
st.title("📈 三角函数拟合工具")

st.markdown("在下方表格中输入 **时间 (s)** 和 **位移 (cm)** 数据，点击按钮进行拟合。")

# ================= Matplotlib 中文字体设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 输入表格 =================
# 默认时间 0 ~ 1.4 (步长 0.1)，位移用一个简单正弦模拟（单位: cm）
default_time = np.round(np.arange(0, 1.5, 0.1), 2)
default_disp = np.round(30 * np.sin(2 * np.pi * default_time), 2)  # 30cm 振幅，示例数据

default_data = pd.DataFrame({
    "时间 (s)": default_time,
    "位移 (cm)": default_disp
})

data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

# 输入的数据
t_data = np.array(data["时间 (s)"])
y_data = np.array(data["位移 (cm)"])

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
            ax.set_xlabel("时间 (s)", fontproperties=myfont)
            ax.set_ylabel("位移 (cm)", fontproperties=myfont)
            ax.legend(prop=myfont)
            ax.set_title("三角函数拟合结果", fontproperties=myfont)
            st.pyplot(fig)

with col2:
    if st.button("🧮 显示函数表达式"):
        params = fit_data(t_data, y_data)
        if params is not None:
            A, omega, phi, C = params
            st.success(
                f"拟合函数:\n\n y(t) = {A:.2f} * sin({omega:.2f} * t + {phi:.2f}) + {C:.2f}   （单位: cm）"
            )


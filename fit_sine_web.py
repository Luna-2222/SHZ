import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st
import matplotlib.font_manager as fm
from io import BytesIO

# ================= 页面/布局 =================
st.set_page_config(page_title="弹簧振子的位移随时间变化规律", layout="wide")
st.title("🔔 弹簧振子的位移随时间变化规律（curve_fit 拟合）")
st.markdown(
    """
    <div style="background-color:#f5f5f7; padding:12px 14px; border-radius:8px;">
      在下方表格中输入 <b>时间 (s)</b> 和 <b>位移 (cm)</b> 数据。点击 <b>显示图片</b> 或 <b>显示公式</b> 进行拟合展示。<br>
      两个按钮互不影响：先后点击均可同时显示图像与公式。
    </div>
    """,
    unsafe_allow_html=True
)

# ================= 仅使用 SimHei.ttf =================
def get_simhei():
    try:
        return fm.FontProperties(fname="SimHei.ttf")
    except Exception:
        st.warning("未找到 SimHei.ttf，已回退为系统字体（可能导致中文不美观）。")
        return fm.FontProperties()
myfont = get_simhei()
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ================= 默认数据 =================
# 时间默认 0~1.40，步长 0.1
default_time = np.round(np.arange(0.0, 1.41, 0.1), 2)

# 示例位移（可在表格中修改），单位 cm
default_disp = np.array(
    [-4.78, -2.40, -1.20,  1.10,  4.10,  5.20,  3.80,  0.60,
     -2.65, -4.90, -4.70, -2.28, -0.70,  2.15,  3.75], dtype=float
)
if default_disp.size != default_time.size:  # 安全兜底
    default_disp = np.round(5 * np.sin(2*np.pi*default_time - 0.8) + 0.5, 2)

default_df = pd.DataFrame({"时间 (s)": default_time, "位移 (cm)": default_disp})

# ================= 数据编辑表格 =================
data = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "时间 (s)": st.column_config.NumberColumn("时间 (s)", step=0.01, help="单位：秒"),
        "位移 (cm)": st.column_config.NumberColumn("位移 (cm)", format="%.2f", step=0.01, help="单位：厘米，保留两位小数"),
    }
)

def read_clean(df: pd.DataFrame):
    t = pd.to_numeric(df["时间 (s)"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["位移 (cm)"], errors="coerce").to_numpy()
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    order = np.argsort(t)
    t, y = t[order], np.round(y[order], 2)
    return t, y

t_data, y_data = read_clean(data)

# ================= 拟合（curve_fit） =================
def sine_model(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

def fit_with_curve_fit(t, y):
    if t.size < 4:
        raise ValueError("数据点太少，至少需要 4 个点。")
    p0 = [(np.max(y) - np.min(y)) / 2.0, 2*np.pi, 0.0, float(np.mean(y))]
    popt, _ = curve_fit(sine_model, t, y, p0=p0, maxfev=20000)
    A, omega, phi, C = popt
    yhat = sine_model(t, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return (A, omega, phi, C), r2

# ================= 状态管理 =================
if "show_plot" not in st.session_state:
    st.session_state.show_plot = False
if "show_formula" not in st.session_state:
    st.session_state.show_formula = False
if "fit_params" not in st.session_state:
    st.session_state.fit_params = None

# ===== 按钮区：显示图片 / 显示公式 / 清除结果 =====
b1, b2, b3 = st.columns([1, 1, 1])
with b1:
    if st.button("🎨 显示图片", use_container_width=True):
        try:
            params, r2 = fit_with_curve_fit(t_data, y_data)
            st.session_state.fit_params = (params, r2)
            st.session_state.show_plot = True      # 打开图片
        except Exception as e:
            st.error(f"拟合失败：{e}")
with b2:
    if st.button("🧮 显示公式", use_container_width=True):
        try:
            params, r2 = fit_with_curve_fit(t_data, y_data)
            st.session_state.fit_params = (params, r2)
            st.session_state.show_formula = True   # 打开公式
        except Exception as e:
            st.error(f"拟合失败：{e}")
with b3:
    if st.button("🧹 清除结果", use_container_width=True):
        st.session_state.show_plot = False
        st.session_state.show_formula = False
        st.session_state.fit_params = None

# ===== 两栏：左公式 / 右图片（互不覆盖，独立显示） =====
left, right = st.columns([1, 2])

# —— 左：公式 —— #
with left:
    if st.session_state.show_formula and st.session_state.fit_params is not None:
        (A, omega, phi, C), r2 = st.session_state.fit_params
        st.markdown("#### 拟合函数表达式")
        st.latex(r"x(t) = %.2f \cdot \sin(%.2f\,t + %.2f) + %.2f" % (A, omega, phi, C))
        st.markdown(f"**R² = {r2:.4f}**（单位：cm）")

# —— 右：图片（含 PNG 下载） —— #
with right:
    if st.session_state.show_plot and st.session_state.fit_params is not None:
        (A, omega, phi, C), r2 = st.session_state.fit_params
        t_fit = np.linspace(t_data.min(), t_data.max(), 500) if t_data.size > 1 else t_data
        y_fit = sine_model(t_fit, A, omega, phi, C)

        fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=300)
        ax.scatter(t_data, y_data, label="实验数据", s=25)
        ax.plot(t_fit, y_fit, label=f"拟合曲线 (R²={r2:.3f})", linewidth=2)
        ax.set_xlabel("时间 (s)", fontproperties=myfont)
        ax.set_ylabel("位移 (cm)", fontproperties=myfont)
        ax.set_title("弹簧振子的位移随时间变化规律", fontproperties=myfont)
        ax.legend(prop=myfont)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        st.pyplot(fig)

        # —— PNG 下载 —— #
        png_buf = BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        png_buf.seek(0)
        st.download_button(
            label="⬇️ 下载PNG",
            data=png_buf,
            file_name="fit_result.png",
            mime="image/png",
            use_container_width=True
        )

# 初始提示
if not st.session_state.show_plot and not st.session_state.show_formula:
    st.info("👉 请输入数据后，点击上方按钮显示图片或公式。")

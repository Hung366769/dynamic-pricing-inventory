"""Streamlit dashboard for data, elasticity, what-if pricing and Q-learning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.demand.elasticity import fit_elasticity
from src.evaluation.baselines import evaluate_policies
from src.environment.inventory_env import ACTIONS

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/processed/retail_price_clean.csv"
RAW_PATH = ROOT / "data/raw/retail_price.csv"
Q_PATH = ROOT / "artifacts/q_table.npy"

st.set_page_config(page_title="Dynamic Pricing + Inventory RL", page_icon="📦", layout="wide")

@st.cache_data
def get_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    elif RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        df["date"] = pd.to_datetime(df["month_year"], format="%d-%m-%Y", errors="coerce")
        comp = [c for c in ["comp_1", "comp_2", "comp_3"] if c in df.columns]
        df["competitor_price"] = df[comp].replace(0, np.nan).mean(axis=1)
    else:
        raise FileNotFoundError("Không tìm thấy CSV. Đặt file tại data/raw/retail_price.csv.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "competitor_price" not in df.columns:
        comp = [c for c in ["comp_1", "comp_2", "comp_3"] if c in df.columns]
        df["competitor_price"] = df[comp].replace(0, np.nan).mean(axis=1)
    return df.dropna(subset=["date", "product_id", "qty", "unit_price", "competitor_price"]).sort_values(["product_id", "date"])

@st.cache_data
def get_elasticity(date_cutoff: str):
    df = get_data()
    _, result, predictions = fit_elasticity(df, train_end=date_cutoff)
    return result, predictions

df = get_data()
page = st.sidebar.radio("Điều hướng", ["Tổng quan dữ liệu", "Price elasticity", "Mô phỏng giá", "Đánh giá Q-learning"])
st.sidebar.caption("Lưu ý: kết quả tồn kho/giá phản thực là mô phỏng, không phải hiệu quả thực tế đã chứng minh.")
cutoff = st.sidebar.date_input("Ngày cuối tập huấn luyện", value=pd.Timestamp("2018-04-01").date(), min_value=df["date"].min().date(), max_value=df["date"].max().date()).isoformat()

if page == "Tổng quan dữ liệu":
    st.title("Dynamic Pricing + Inventory Management with RL")
    st.write("Dashboard nghiên cứu cho Retail Price Optimization. Dữ liệu tồn kho và giá vốn không có sẵn trong dataset gốc; các kết quả này cần giả định hoặc dữ liệu bổ sung.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Số dòng", f"{len(df):,}")
    c2.metric("Sản phẩm", f"{df.product_id.nunique():,}")
    c3.metric("Số tháng", f"{df.date.dt.to_period('M').nunique():,}")
    c4.metric("Khoảng thời gian", f"{df.date.min():%Y-%m} → {df.date.max():%Y-%m}")
    st.subheader("Giá và lượng bán theo thời gian")
    product = st.selectbox("Chọn sản phẩm", sorted(df.product_id.unique()))
    g = df[df.product_id == product].sort_values("date")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(g.date, g.unit_price, marker="o", label="Giá bán")
    ax.set_ylabel("Giá bán")
    ax2 = ax.twinx(); ax2.plot(g.date, g.qty, marker="s", linestyle="--", label="Lượng bán")
    ax2.set_ylabel("Lượng bán")
    ax.set_title(f"{product}: giá và lượng bán lịch sử")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("Các cột chính")
    st.dataframe(df[[c for c in ["product_id","date","unit_price","qty","total_price","competitor_price","lag_price","product_score"] if c in df.columns]].head(30), use_container_width=True)

elif page == "Price elasticity":
    st.title("Ước lượng price elasticity")
    with st.spinner("Đang ước lượng mô hình log-log..."):
        result, preds = get_elasticity(cutoff)
    e = result["elasticity_estimate"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Ước lượng elasticity", f"{e:.3f}")
    c2.metric("95% CI", f"[{result['elasticity_ci95_low']:.2f}, {result['elasticity_ci95_high']:.2f}]")
    c3.metric("p-value (cluster theo sản phẩm)", f"{result['elasticity_p_value_clustered_by_product']:.4g}")
    c4.metric("R² train", f"{result['r_squared_train']:.3f}")
    st.markdown("**Mô hình:** `log(qty) ~ log(unit_price) + log(competitor_price) + product fixed effects + month fixed effects + year fixed effects`.")
    st.json({k:v for k,v in result.items() if k not in ["formula","interpretation_warning"]})
    st.warning("Hệ số này là mối liên hệ có điều kiện trong dữ liệu quan sát, chưa đủ để kết luận quan hệ nhân quả. Giá có thể phụ thuộc vào khuyến mãi hoặc kỳ vọng nhu cầu. MAPE có thể gây hiểu nhầm khi qty nhỏ; nên xem MAE, RMSE, WAPE và sMAPE.")
    if not preds.empty:
        st.subheader("Dự báo lượng bán trên giai đoạn thời gian giữ lại")
        st.dataframe(preds.head(100), use_container_width=True)

elif page == "Mô phỏng giá":
    st.title("Mô phỏng kịch bản giá")
    st.caption("Đây là what-if simulation: lượng cầu phản thực được tính bằng qty quan sát × (giá mới/giá quan sát)^elasticity.")
    result, _ = get_elasticity(cutoff)
    elasticity = result["elasticity_estimate"]
    products = sorted(df.product_id.unique())
    product = st.selectbox("Sản phẩm", products)
    g = df[df.product_id == product].sort_values("date")
    row = g.iloc[-1]
    base_price = float(row.unit_price)
    base_qty = float(row.qty)
    competitor = float(row.competitor_price)
    cost_ratio = st.slider("Giả định giá vốn / giá trung vị", 0.30, 0.95, 0.70, 0.05)
    inventory = st.slider("Tồn kho hiện tại (đơn vị)", 1, 500, max(1, int(round(base_qty*3))))
    horizon = st.slider("Số kỳ còn lại (cho trạng thái Q-learning)", 1, 6, 4)
    holding_rate = st.slider("Chi phí lưu kho / giá vốn / đơn vị tồn", 0.0, 0.10, 0.01, 0.005)
    stockout_penalty = st.slider("Phạt thiếu hàng / giá vốn / đơn vị thiếu", 0.0, 1.0, 0.25, 0.05)
    med_price = float(g.unit_price.median())
    unit_cost = med_price * cost_ratio
    rows=[]
    for action in ACTIONS:
        price = base_price * (1+action)
        demand = max(0, int(round(base_qty * (price/base_price)**elasticity)))
        sold = min(demand, inventory)
        end_inv = inventory - sold
        revenue = price * sold
        profit = (price-unit_cost)*sold - holding_rate*unit_cost*end_inv - stockout_penalty*unit_cost*max(0,demand-sold)
        rows.append({"Điều chỉnh giá": f"{action:+.0%}", "Giá kịch bản": price, "Cầu mô phỏng": demand, "Bán được": sold, "Tồn cuối kỳ": end_inv, "Doanh thu mô phỏng": revenue, "Lợi nhuận sau phạt mô phỏng": profit})
    scen = pd.DataFrame(rows)
    best = scen.iloc[scen["Lợi nhuận sau phạt mô phỏng"].argmax()]
    c1,c2,c3 = st.columns(3)
    c1.metric("Giá tham chiếu", f"{base_price:,.2f}")
    c2.metric("Giá đề xuất theo grid", f"{best['Giá kịch bản']:,.2f}", f"{best['Điều chỉnh giá']}")
    c3.metric("Lợi nhuận mô phỏng", f"{best['Lợi nhuận sau phạt mô phỏng']:,.2f}")
    st.dataframe(scen, use_container_width=True)
    st.caption(f"Giá đối thủ trung bình gần nhất: {competitor:,.2f}; elasticity dùng trong mô phỏng: {elasticity:.3f}. Giá vốn và tồn kho là tham số nhập thủ công.")
    if Q_PATH.exists():
        q_table = np.load(Q_PATH)
        med_qty = max(float(g.qty.median()), 1.0)
        forecast = float(g.qty.tail(3).mean())
        inventory_reference = max(float(inventory), med_qty * horizon)
        inv_ratio = inventory / max(inventory_reference, 1.0)
        inv_bin = 0 if inv_ratio < 0.33 else (2 if inv_ratio > 0.67 else 1)
        demand_ratio = forecast / med_qty
        demand_bin = 0 if demand_ratio < 0.80 else (2 if demand_ratio > 1.20 else 1)
        price_ratio = base_price / max(competitor, 1e-6)
        price_bin = 0 if price_ratio < 0.97 else (2 if price_ratio > 1.03 else 1)
        time_ratio = horizon / 6.0
        time_bin = 2 if time_ratio > 0.66 else (1 if time_ratio > 0.33 else 0)
        state = (inv_bin, demand_bin, price_bin, time_bin)
        action_idx = int(np.argmax(q_table[state]))
        q_price = float(np.clip(base_price*(1+ACTIONS[action_idx]), med_price*0.70, med_price*1.30))
        q_demand = max(0, int(round(base_qty*(q_price/base_price)**elasticity)))
        q_sold = min(q_demand, inventory)
        q_end_inv = inventory - q_sold
        q_profit = (q_price-unit_cost)*q_sold - holding_rate*unit_cost*q_end_inv - stockout_penalty*unit_cost*max(0, q_demand-q_sold)
        st.subheader("Khuyến nghị từ Q-learning đã huấn luyện")
        q1,q2,q3 = st.columns(3)
        q1.metric("Hành động Q-learning", f"{ACTIONS[action_idx]:+.0%}")
        q2.metric("Giá được đề xuất", f"{q_price:,.2f}")
        q3.metric("Lợi nhuận 1 kỳ mô phỏng", f"{q_profit:,.2f}")
        st.warning("Q-table được huấn luyện với bộ tham số mặc định (giá vốn 70% giá trung vị, chi phí lưu kho 1%, phạt thiếu hàng 25%, phạt đổi giá 5%, không nhập thêm hàng). Thay đổi các thanh trượt sẽ cập nhật bảng what-if nhưng không tự huấn luyện lại chính sách Q-learning.")
    else:
        st.info("Chưa có Q-table. Chạy `python train.py` để huấn luyện và bật đề xuất từ Q-learning.")
    fig, ax = plt.subplots(figsize=(8,4)); ax.bar(scen["Điều chỉnh giá"],scen["Lợi nhuận sau phạt mô phỏng"]); ax.set_ylabel("Lợi nhuận mô phỏng"); ax.set_title("So sánh hành động giá"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

else:
    st.title("Đánh giá chính sách")
    st.write("Chạy `python train.py` để tạo Q-table; sau đó chạy `python evaluate.py` để đánh giá trên giai đoạn giữ lại.")
    summary_path = ROOT / "outputs/policy_evaluation_summary.csv"
    reward_path = ROOT / "outputs/training_rewards.csv"
    if reward_path.exists():
        rewards = pd.read_csv(reward_path)
        fig, ax = plt.subplots(figsize=(10,4)); ax.plot(rewards.episode, rewards.total_reward, alpha=.35, label="Reward từng episode")
        if len(rewards) >= 50:
            ax.plot(rewards.episode, rewards.total_reward.rolling(50).mean(), label="Trung bình trượt 50")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward mô phỏng"); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Chưa có bảng đánh giá. Hãy huấn luyện và chạy evaluate.py trước.")
    if Q_PATH.exists():
        st.success("Đã tìm thấy Q-table đã huấn luyện.")
    else:
        st.warning("Chưa tìm thấy artifacts/q_table.npy.")

# Dynamic Pricing + Inventory Management with Q-learning

Dự án nghiên cứu sử dụng dataset **Retail Price Optimization** để (1) ước lượng price elasticity, (2) xây dựng mô phỏng quyết định giá kết hợp tồn kho, (3) huấn luyện Q-learning, (4) so sánh chính sách và (5) trình bày kết quả trên Streamlit.

> **Phân biệt bắt buộc:** giá bán, lượng bán, giá đối thủ và các trường khác trong CSV là dữ liệu quan sát. Tồn kho ban đầu, giá vốn, chi phí lưu kho, chi phí thiếu hàng, lượng nhập và nhu cầu phản thực ở mức giá chưa từng áp dụng là giả định/mô phỏng nếu không có nguồn dữ liệu bổ sung. Kết quả RL trong dự án này là kết quả trên simulator, không phải bằng chứng hiệu quả trong kinh doanh thực tế.

## 1. Cấu trúc

- `data/raw/retail_price.csv`: CSV được cung cấp.
- `data/processed/retail_price_clean.csv`: dữ liệu đã parse ngày, chuẩn hóa kiểu số và sắp xếp theo sản phẩm/thời gian.
- `src/data/preprocess.py`: làm sạch dữ liệu.
- `src/data/eda.py`: báo cáo EDA và hình ảnh.
- `src/demand/elasticity.py`: ước lượng elasticity bằng hồi quy log-log có product/month/year fixed effects, SE cluster theo sản phẩm, và đánh giá theo thời gian.
- `src/environment/inventory_env.py`: historical-replay simulator cho giá/tồn kho; hỗ trợ đặt hàng EOQ + ROP với lead-time queue.
- `src/inventory_policies.py`: hàm tính EOQ, ROP và inventory position.
- `src/rl/q_learning.py`: Q-learning tabular.
- `src/evaluation/baselines.py`: so sánh chính sách.
- `train.py`: huấn luyện tác nhân.
- `evaluate.py`: đánh giá trên giai đoạn giữ lại.
- `app.py`: Streamlit dashboard.

## 2. Cài đặt

Khuyến nghị Python 3.10–3.12. Tại thư mục `dynamic_pricing_rl_project`:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell không cho phép activate môi trường, có thể chạy trực tiếp `.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## 3. Chạy theo đúng thứ tự

```bash
python src/data/preprocess.py
python -m src.data.eda
python -m src.demand.elasticity --train-end 2018-04-01
python train.py
python evaluate.py
python run_experiments.py --episodes 1000 --seeds 42,7,21,2025,2026
python -m src.evaluation.plot_results
streamlit run app.py
```

- Tập huấn luyện mặc định kết thúc **2018-04-01** (bao gồm ngày đó).
- Tập kiểm thử thời gian gồm **2018-05 đến 2018-08**.
- Mặc định huấn luyện 3.000 episodes. Có thể chạy `python -m src.rl.q_learning --episodes 1000` để thử nhanh.
- `train.py` tạo `artifacts/q_table.npy`, cấu hình huấn luyện và lịch sử reward.
- `evaluate.py` lưu `outputs/policy_evaluation_summary.csv` và `outputs/policy_evaluation_detail.csv`.
- `run_experiments.py` lưu kết quả từng seed, bảng mean/std và thống kê reward cuối quá trình huấn luyện.

## 4. Phương pháp ước lượng price elasticity

Mô hình chính:

\[
\log(Q_{it}) = \beta_0 + \beta_1\log(P_{it}) + \beta_2\log(P^{comp}_{it}) + \alpha_i + \delta_{month} + \tau_{year} + \epsilon_{it}
\]

Trong đó `Q` là `qty`, `P` là `unit_price`, `P_comp` là trung bình các cột competitor price hợp lệ, `alpha_i` là product fixed effects. Với mô hình log-log, `beta_1` được diễn giải là phần trăm thay đổi lượng bán liên hệ với 1% thay đổi giá, giữ các biến có trong mô hình không đổi.

**Giới hạn:** đây là dữ liệu quan sát. Giá có thể được chọn theo kỳ vọng nhu cầu, khuyến mãi hoặc thông tin không quan sát; vì thế hệ số không mặc nhiên là độ co giãn nhân quả. Mô hình lượng bán trên tập giữ lại cũng cần kiểm tra kỹ; nếu sai số còn lớn, không nên coi dự báo cầu đã sẵn sàng cho triển khai thực tế.

## 5. Định nghĩa MDP

### State

`(inventory_bucket, demand_forecast_bucket, price_vs_competitor_bucket, time_remaining_bucket)`, mỗi biến có 3 mức. Tổng cộng tối đa 81 trạng thái.

- Tồn kho: thấp/trung bình/cao theo tỷ lệ tồn kho so với mức tham chiếu.
- Dự báo nhu cầu: thấp/trung bình/cao theo lượng bán gần đây so với median lượng bán của sản phẩm.
- Vị thế giá: thấp/tương đương/cao hơn giá đối thủ.
- Thời gian còn lại: đầu/giữa/cuối episode.

### Action

`[-10%, -5%, 0%, +5%, +10%]`; giá bị giới hạn trong khoảng ±30% quanh giá trung vị của sản phẩm trong tập huấn luyện.

### Reward

\[
r_t=(p_t-c)q_t-hI_{t+1}-bU_t-\lambda\,\Delta p_t
\]

Trong code, mỗi thành phần được triển khai cụ thể: lợi nhuận gộp, phạt lưu kho, phạt nhu cầu không đáp ứng và phạt thay đổi giá. `c`, `h`, `b`, và `lambda` là tham số mô phỏng, không có sẵn đầy đủ trong CSV.

### Cập nhật Q-learning

\[
Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]
\]

Thiết lập khởi đầu: `alpha=0.1`, `gamma=0.95`, epsilon-greedy, epsilon giảm dần, 3.000 episodes. Cần báo cáo cả đường reward theo episode và độ ổn định qua seed, không chỉ chọn episode có kết quả cao nhất.

## 6. Các giả định mặc định của simulator

Các giá trị sau chỉ là kịch bản khởi đầu, cần làm phân tích độ nhạy:

- Giá vốn đơn vị = 70% giá trung vị lịch sử của sản phẩm.
- Tồn kho đầu kỳ = lượng bán trung vị × số tháng tồn kho giả định (trong kiểm thử mặc định 2,5 tháng).
- Chi phí lưu kho = 1% giá vốn trên mỗi đơn vị tồn cuối kỳ.
- Phạt thiếu hàng = 25% giá vốn trên mỗi đơn vị cầu bị thiếu.
- Phạt thay đổi giá = tỷ lệ 5% được nhân với quy mô giá vốn và lượng bán trung vị.
- Bổ sung hàng mặc định bằng 0; có thể thay đổi tham số trong code.
- Cầu phản thực tại giá mới được mô phỏng bằng `qty_observed * (new_price / observed_unit_price) ** elasticity`, với nhiễu trong giai đoạn training và không có nhiễu trong so sánh chính sách để bảo đảm tái lập.

`qty_observed` của tháng trong historical replay đóng vai trò **mức cầu nền đã quan sát ở giá thực tế**, không được cung cấp cho tác nhân trước khi chọn giá. Dù vậy, đây vẫn là phép mô phỏng phản thực đơn giản; nó giả định cùng cú sốc cầu nền sẽ xuất hiện ở mức giá mới và áp dụng elasticity ước lượng cho các mức giá được thử. Vì vậy, không diễn giải chênh lệch lợi nhuận mô phỏng như mức tăng lợi nhuận thật.

## 7. Baselines

- `fixed_price`: giữ mức giá ban đầu.
- `random_policy`: chọn ngẫu nhiên một trong năm hành động giá.
- `match_competitor`: chọn một trong các hành động để tiến gần giá đối thủ.
- `inventory_rule`: giảm giá khi tồn kho cao, tăng nhẹ khi tồn kho thấp, giữ giá khi ở giữa.
- `fixed_qty_rop`: đặt hàng theo ROP với lượng đặt cố định (2 × median monthly demand).
- `eoq_rop`: đặt hàng theo ROP với lượng đặt bằng EOQ.
- `q_learning`: hành động greedy theo Q-table đã huấn luyện. Trong môi trường train, inventory policy mặc định là EOQ + ROP.

Các chính sách được đánh giá trên cùng các sản phẩm/tháng kiểm thử và cùng giả định về giá vốn, tồn kho và cầu nền.

## 8. Câu hỏi nghiên cứu gợi ý

1. Mối liên hệ giá–lượng bán quan sát được là bao nhiêu sau khi kiểm soát khác biệt cố hữu giữa sản phẩm và mùa vụ?
2. Trong simulator được xây dựng, Q-learning có cải thiện tổng reward so với các baseline không?
3. Kết quả thay đổi ra sao khi thay đổi elasticity, tồn kho đầu kỳ, tỷ lệ giá vốn, chi phí lưu kho và mức phạt thiếu hàng?

## 9. Tài liệu tham khảo khởi đầu

1. Chenavaz, R. Y., & Dimitrov, S. (2025). *Artificial intelligence and dynamic pricing: a systematic literature review*. Journal of Applied Economics. https://doi.org/10.1080/15140326.2025.2466140
2. *Dynamic pricing of differentiated products with incomplete information based on reinforcement learning* (2022), IET Collaborative Intelligent Manufacturing. https://doi.org/10.1049/cim2.12050
3. *Dynamic Pricing and Learning: Historical Origins, Current Research, and New Directions* (2015), Surveys in Operations Research and Management Science. https://doi.org/10.1016/j.sorms.2015.03.001
4. Public CSV schema cross-check: https://github.com/ayush714/retail-price-optimization/blob/main/data/retail_price.csv

## 10. Những gì chưa được khẳng định

- Chưa có dữ liệu tồn kho thực theo từng sản phẩm/tháng.
- Chưa có giá vốn thực tế.
- Chưa xác lập quan hệ nhân quả giữa giá và lượng bán.
- Kết quả Q-learning chỉ phản ánh môi trường mô phỏng hiện tại; cần kiểm thử độ nhạy và nhiều random seeds trước khi kết luận nghiên cứu.


## 11. EOQ + ROP replenishment extension

The inventory simulator now supports an EOQ order quantity and an ROP trigger.
Demand and lead time in the ROP calculation use monthly periods in this replay
experiment. EOQ annualizes the training-period median monthly demand.

\[
Q^* = \sqrt{2DS/H}, \qquad ROP = dL + SS
\]

- `D`: annualized demand (median monthly training demand × 12).
- `S`: assumed ordering cost per order (default 25 currency units).
- `H`: assumed annual holding cost per unit (default 20% of assumed unit cost).
- `d`: median monthly demand; `L`: lead time in months; `SS`: safety stock units.
- Inventory position includes on-hand stock plus outstanding orders.

The evaluation now reports `fixed_qty_rop` and `eoq_rop` side by side. Run:

```bash
python evaluate.py --ordering-cost 25 --annual-holding-cost-rate 0.20 --lead-time-periods 1 --safety-stock-units 0
```

All four EOQ/ROP inputs are illustrative scenario assumptions, not fields observed
in the Retail Price Optimization dataset. For the most interpretable comparison,
keep the same assumptions across policies and run sensitivity tests (for example,
vary ordering cost, holding-cost rate, lead time, and safety stock). The EOQ+ROP
policies in `evaluate.py` use the same fixed price path to isolate replenishment
effects; Q-learning remains available as a separate pricing policy. In the training
environment, the Q-learning agent is trained with the configured EOQ+ROP policy.

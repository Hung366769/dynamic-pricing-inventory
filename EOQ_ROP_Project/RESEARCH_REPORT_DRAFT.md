# Bản thảo nội dung nghiên cứu

## Tên đề tài

**Xây dựng mô hình hỗ trợ ra quyết định định giá động kết hợp quản lý tồn kho bằng học tăng cường và độ co giãn của cầu theo giá**

## Tóm tắt dự kiến

Nghiên cứu đề xuất một quy trình hỗ trợ ra quyết định định giá động trong đó độ co giãn của cầu theo giá được ước lượng từ dữ liệu bán lẻ lịch sử và được sử dụng trong môi trường mô phỏng cho tác nhân Q-learning. Dataset Retail Price Optimization cung cấp quan sát theo sản phẩm và tháng về giá bán, số lượng bán, giá đối thủ và các đặc trưng thời gian. Mô hình kinh tế lượng log-log với hiệu ứng cố định theo sản phẩm và lịch được sử dụng để ước lượng mối liên hệ giữa giá và lượng bán. Sau đó, một môi trường mô phỏng kết hợp hành động điều chỉnh giá, tồn kho đầu kỳ, lượng bán, hàng thiếu và chi phí tồn kho được xây dựng để huấn luyện Q-learning. Mô hình được so sánh với một số chính sách định giá cơ sở trên giai đoạn thời gian giữ lại. Do dữ liệu gốc không có tồn kho và giá vốn đầy đủ, kết quả của phần RL được báo cáo là kết quả mô phỏng có điều kiện theo các giả định, không phải ước lượng tác động kinh doanh thực tế.

## 1. Bối cảnh và vấn đề nghiên cứu

Định giá động là bài toán quyết định giá theo thời gian dựa trên thông tin về nhu cầu, cạnh tranh và trạng thái vận hành. Nếu chỉ tối ưu doanh thu ngắn hạn, nhà bán lẻ có thể bán được nhiều hàng nhưng lợi nhuận thấp do giảm giá sâu, hoặc đối mặt với tình trạng hết hàng. Ngược lại, giá quá cao có thể làm lượng bán thấp và tăng tồn kho cuối kỳ.

Độ co giãn giá của cầu cung cấp một cách định lượng để mô tả mức thay đổi của lượng cầu khi giá thay đổi. Tuy nhiên, hệ số ước lượng từ dữ liệu quan sát có thể bị ảnh hưởng bởi yếu tố mùa vụ, khác biệt giữa sản phẩm, khuyến mãi và việc doanh nghiệp chủ động thay đổi giá khi dự báo nhu cầu khác nhau. Vì vậy, kết quả hồi quy cần được xem là mối liên hệ có điều kiện trừ khi nghiên cứu có thiết kế nhận dạng nhân quả phù hợp.

Q-learning là một phương pháp học tăng cường dạng giá trị có thể học chính sách ra quyết định qua các tương tác với môi trường mô phỏng. Trong nghiên cứu này, Q-learning được dùng để lựa chọn một trong các hành động điều chỉnh giá, với phần thưởng phản ánh lợi nhuận và chi phí liên quan đến tồn kho, thiếu hàng và biến động giá.

## 2. Mục tiêu nghiên cứu

1. Khảo sát và tiền xử lý dữ liệu bán lẻ theo sản phẩm-thời gian.
2. Ước lượng hệ số liên hệ giá–lượng bán bằng mô hình log-log có kiểm soát khác biệt sản phẩm và mùa vụ.
3. Xây dựng môi trường mô phỏng định giá và tồn kho với các giả định được ghi nhận rõ.
4. Huấn luyện Q-learning và so sánh với chính sách giá cố định, chính sách ngẫu nhiên, bám giá đối thủ và quy tắc tồn kho.
5. Phân tích độ nhạy đối với elasticity, tồn kho ban đầu, giá vốn và các khoản phạt.
6. Xây dựng dashboard cho phép xem dữ liệu, hệ số elasticity và các kịch bản định giá.

## 3. Câu hỏi nghiên cứu

- **RQ1:** Sau khi kiểm soát hiệu ứng sản phẩm và thời gian, hệ số log-log giữa giá và lượng bán trong dataset có dấu và độ lớn như thế nào?
- **RQ2:** Trong môi trường mô phỏng đã công bố rõ các giả định, Q-learning có đạt tổng reward cao hơn các chính sách cơ sở không?
- **RQ3:** Chính sách Q-learning thay đổi thế nào theo mức tồn kho, độ co giãn cầu và chi phí lưu kho/thiếu hàng?

## 4. Dữ liệu

### 4.1 Nguồn và quy mô

File được cung cấp có 676 quan sát và 30 cột gốc, gồm 52 mã sản phẩm thuộc 9 nhóm danh mục, từ tháng 01/2017 đến tháng 08/2018. Dữ liệu là panel không cân bằng: số quan sát và thời điểm bắt đầu/kết thúc khác nhau giữa các sản phẩm.

### 4.2 Các biến chính

| Biến | Ý nghĩa sử dụng |
|---|---|
| `product_id` | Định danh sản phẩm; dùng cho hiệu ứng cố định và chuỗi thời gian từng sản phẩm |
| `month_year` | Thời điểm quan sát; parse thành ngày để sắp xếp và chia train/test |
| `qty` | Lượng bán quan sát; biến phụ thuộc trong hồi quy log-log |
| `unit_price` | Giá bán quan sát; biến giá chính |
| `comp_1`, `comp_2`, `comp_3` | Giá của các đối thủ được ghi nhận; dùng trung bình làm biến cạnh tranh |
| `lag_price` | Giá trễ; kiểm tra tính nhất quán với giá kỳ trước |
| `holiday`, `month`, `year` | Yếu tố thời gian; có thể dùng cho phân tích mùa vụ |
| `total_price` | Trường tổng giá trị trong dữ liệu; không tự động thay thế `qty × unit_price` khi chưa xác minh định nghĩa |

### 4.3 Kết quả kiểm tra dữ liệu

- 676 dòng được giữ lại sau khi kiểm tra các trường khóa; không có giá trị trống trong các trường CSV gốc ở file này.
- Không phát hiện khóa `product_id`–tháng trùng.
- `qty` nằm trong khoảng 1–122; median bằng 10.
- `unit_price` nằm trong khoảng 19.90–364.00; median bằng 89.90.
- 50/52 sản phẩm thay đổi giá trong toàn bộ khoảng thời gian; 48/52 sản phẩm thay đổi giá trong tập huấn luyện mặc định.
- Có 595 cặp tháng liên tiếp cùng sản phẩm; `lag_price` khớp giá `unit_price` kỳ trước với sai lệch không quá 0.03 ở cả 595 cặp này.
- `total_price` không bằng `qty × unit_price` trong 79 dòng với sai lệch tuyệt đối lớn hơn 0.01. Cần giữ nguyên trường này và xác minh định nghĩa thay vì tự tính lại doanh thu bằng một trong hai biểu thức.

### 4.4 Chia dữ liệu theo thời gian

- **Train:** đến hết ngày 01/04/2018; 516 dòng.
- **Test:** từ 01/05/2018 đến 01/08/2018; 160 dòng.

Chia theo thời gian giúp hạn chế rò rỉ dữ liệu khi mô phỏng quyết định ở tương lai. Tuy nhiên, giai đoạn test ngắn nên độ ổn định của các chỉ số còn hạn chế.

## 5. Phương pháp ước lượng elasticity

Mô hình cơ sở:

\[
\log(Q_{it}) = \beta_0 + \beta_1\log(P_{it}) + \beta_2\log(P^{comp}_{it}) + \alpha_i + \delta_m + \tau_y + \epsilon_{it}
\]

Trong đó `Q` là `qty`, `P` là `unit_price`, `P_comp` là trung bình các giá đối thủ khả dụng, `alpha_i` là hiệu ứng cố định sản phẩm, `delta_m` là hiệu ứng tháng trong năm và `tau_y` là hiệu ứng năm. Sai số chuẩn được cluster theo sản phẩm.

Với dạng log-log, `beta_1` là hệ số co giãn có điều kiện: một thay đổi 1% của giá liên hệ với thay đổi khoảng `beta_1%` của lượng bán, khi các biến được giữ cố định. Hệ số này không nên mặc định được diễn giải theo quan hệ nhân quả.

### Kết quả sơ bộ trên train

Với tập train kết thúc ngày 01/04/2018, mô hình hiện tại cho:

- Hệ số `log_price`: **-2.856**.
- Khoảng tin cậy 95% với sai số chuẩn cluster: **[-4.146, -1.566]**.
- p-value xấp xỉ **0.0000143**.
- R² trong tập train: **0.403**.

Hệ số âm và độ lớn tuyệt đối lớn hơn 1 gợi ý rằng quan hệ quan sát được trong mô hình cơ sở là co giãn. Tuy nhiên, điều này chưa chứng minh cầu có độ co giãn nhân quả như vậy, bởi giá có thể nội sinh và dữ liệu không có đầy đủ biến khuyến mãi/cú sốc cầu.

### Kiểm tra khả năng dự báo

Khi hoàn nguyên dự báo log-lượng sang thang đo lượng bán có hiệu chỉnh smearing, kết quả giữ lại hiện tại là:

- MAE khoảng **27.55** đơn vị.
- RMSE khoảng **36.14** đơn vị.
- WAPE khoảng **206.74%**.
- sMAPE khoảng **108.70%**.

Để so sánh, baseline dự báo bằng median `qty` của từng sản phẩm trong train có MAE khoảng **11.33**, RMSE khoảng **16.90** và WAPE khoảng **84.99%** trên cùng test. Như vậy, mô hình log-log hiện tại chưa đủ tốt để dùng như một bộ dự báo số lượng độc lập. Cần xem đây là mô hình ước lượng mối liên hệ elasticity trước mắt, và phải cải thiện/đánh giá riêng mô hình dự báo nhu cầu trước khi triển khai thực tế.

## 6. Thiết kế môi trường RL

### 6.1 State

State rời rạc gồm:

1. Mức tồn kho: thấp/trung bình/cao.
2. Mức dự báo cầu: thấp/trung bình/cao, dựa trên tối đa ba quan sát lượng bán trước kỳ hiện tại so với median sản phẩm.
3. Vị thế giá so với đối thủ: thấp/tương đương/cao.
4. Thời gian còn lại: đầu/giữa/cuối episode.

Không gian trạng thái có tối đa `3 × 3 × 3 × 3 = 81` trạng thái.

### 6.2 Action

Năm hành động: giảm 10%, giảm 5%, giữ nguyên, tăng 5%, tăng 10%. Giá bị giới hạn trong khoảng ±30% quanh giá trung vị sản phẩm của tập train.

### 6.3 Hàm phần thưởng

\[
r_t = (p_t-c)q_t - hI_{t+1} - bU_t - \lambda\Delta p_t
\]

Trong đó `p` là giá chọn, `c` là giá vốn giả định, `q` là lượng bán mô phỏng, `I` là tồn kho cuối kỳ, `U` là lượng cầu không đáp ứng, và `Delta p` thể hiện mức thay đổi giá. Các hệ số `c`, `h`, `b`, `lambda` phải được xem là tham số kịch bản khi chưa có dữ liệu chi phí.

### 6.4 Chuyển trạng thái và cầu phản thực

Môi trường historical replay sử dụng `qty` đã quan sát trong tháng như mức cầu nền ở giá thực tế, sau đó mô phỏng cầu ở giá mới bằng:

\[
\hat{Q}_t(p) = Q_t^{obs}\left(\frac{p}{P_t^{obs}}\right)^{\hat{\beta}_1}
\]

Trong training có thêm nhiễu ngẫu nhiên nhân để mô phỏng độ bất định. Tác nhân không được cung cấp `qty` của kỳ đang quyết định trước khi chọn hành động; nó dùng lịch sử lượng bán trước đó để phân loại nhu cầu. Tuy vậy, công thức vẫn là giả định phản thực mạnh: nó giả định lượng cầu nền quan sát được sẽ giữ nguyên nếu đổi giá và hệ số elasticity giữ ổn định trong khoảng giá thử nghiệm.

### 6.5 Thuật toán

Q-learning dùng cập nhật:

\[
Q(s,a) \leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
\]

Khởi điểm: learning rate 0.1, discount factor 0.95, epsilon-greedy, epsilon giảm dần, 3.000 episodes. Cần chạy nhiều seed và kiểm tra moving-average reward để tránh kết luận chỉ từ một lần chạy.

## 7. Đánh giá

So sánh các chính sách:

- Giá cố định.
- Chính sách ngẫu nhiên.
- Chọn hành động đưa giá gần đối thủ.
- Quy tắc theo tồn kho.
- Q-learning.

Chỉ số: doanh thu mô phỏng, lợi nhuận gộp mô phỏng, tổng reward sau các khoản phạt, lượng bán, tồn kho cuối kỳ, cầu không đáp ứng, tỷ lệ hết hàng và mức biến động giá.

### Kết quả chạy thử hiện tại

Đã chạy thử 5 seed huấn luyện độc lập (`42, 7, 21, 2025, 2026`), mỗi seed 1.000 episodes, và đánh giá trên cùng tập test/kịch bản mô phỏng. Kết quả trung bình tổng reward:

| Chính sách | Tổng reward mô phỏng trung bình | Độ lệch chuẩn giữa seed | Doanh thu mô phỏng trung bình | Lợi nhuận gộp mô phỏng trung bình | Tỷ lệ thiếu hàng mô phỏng |
|---|---:|---:|---:|---:|---:|
| Q-learning | 14.017,97 | 3.066,16 | 113.131,37 | 29.965,90 | 35,22% |
| Quy tắc tồn kho | 12.975,82 | 0,00* | 116.774,39 | 29.688,10 | 34,70% |
| Giá cố định | 9.573,97 | 0,00* | 116.402,78 | 29.572,03 | 40,41% |
| Ngẫu nhiên | 4.861,19 | 0,00* | 116.082,04 | 26.773,57 | 42,56% |
| Bám giá đối thủ | -13.782,60 | 0,00* | 112.706,26 | 23.183,40 | 52,15% |

`*` Các baseline không học ngẫu nhiên trong lần đánh giá này nên cho cùng kết quả giữa các seed. Số liệu chỉ thuộc simulator, đơn vị tiền tệ theo dataset chưa được xác minh là gì.

Trong 5 seed, Q-learning có tổng reward trung bình cao hơn quy tắc tồn kho khoảng 8%, nhưng độ lệch chuẩn của Q-learning (khoảng 3.066) lớn hơn khoảng cách trung bình giữa hai chính sách (khoảng 1.042). Vì vậy, kết quả hiện tại **chưa đủ để khẳng định mức cải thiện ổn định**. Q-learning có lợi nhuận gộp mô phỏng nhỉnh hơn một chút nhưng doanh thu thấp hơn; tỷ lệ thiếu hàng cũng không thấp hơn baseline tồn kho. Số reward cao hơn phản ánh tổng thể cấu trúc reward hiện tại, vốn bao gồm phạt thiếu hàng, lưu kho và đổi giá; không nên chỉ dựa vào một chỉ số để kết luận chính sách tốt hơn.

Cần tăng số seed, thay đổi các giả định tồn kho/giá vốn/elasticity, và kiểm tra xem thứ hạng chính sách có duy trì hay không trước khi đưa ra kết luận cuối cùng.

## 8. Giới hạn và hướng cải thiện

1. Thiếu tồn kho lịch sử, giá vốn, lịch nhập hàng và dữ liệu hết hàng.
2. Hệ số elasticity có thể bị nội sinh; không nên gọi đây là tác động nhân quả.
3. Mô hình cầu hiện tại chưa dự báo tốt hơn baseline median trên giai đoạn giữ lại.
4. Replay dùng `qty` test làm cầu nền phản thực, nên đánh giá chính sách vẫn phụ thuộc vào giả định mô phỏng.
5. Q-table có giới hạn khái quát khi tăng số trạng thái/biến sản phẩm.
6. Cần chạy sensitivity analysis cho elasticity, cost ratio, tồn kho đầu kỳ, holding cost, stockout penalty và replenishment rate.
7. Hướng nâng cấp có thể gồm demand model riêng theo sản phẩm/nhóm hàng, phương pháp panel/causal tốt hơn nếu có biến công cụ/thí nghiệm, và RL với nhiều kỳ/nguồn cung thực tế hơn.

## 9. Kết luận sơ bộ

Dataset có đủ cấu trúc để làm nghiên cứu ban đầu về mối liên hệ giá–lượng bán và thử nghiệm Q-learning trên môi trường mô phỏng. Điểm đóng góp khả thi là kết hợp một hệ số elasticity ước lượng từ dữ liệu với trạng thái tồn kho và hàm phần thưởng đa mục tiêu. Tuy nhiên, bằng chứng hiện tại chỉ hỗ trợ kết luận về hiệu quả trong môi trường giả lập. Để đưa ra kết luận kinh doanh thực tế, cần dữ liệu tồn kho/giá vốn và cách xác định nhu cầu phản thực đáng tin cậy hơn.

## Tài liệu tham khảo gợi ý

1. Chenavaz, R. Y., & Dimitrov, S. (2025). *Artificial intelligence and dynamic pricing: a systematic literature review*. Journal of Applied Economics. https://doi.org/10.1080/15140326.2025.2466140
2. *Dynamic pricing of differentiated products with incomplete information based on reinforcement learning*. (2022). IET Collaborative Intelligent Manufacturing. https://doi.org/10.1049/cim2.12050
3. *Dynamic Pricing and Learning: Historical Origins, Current Research, and New Directions*. (2015). Surveys in Operations Research and Management Science. https://doi.org/10.1016/j.sorms.2015.03.001
4. Dataset schema cross-check: https://github.com/ayush714/retail-price-optimization/blob/main/data/retail_price.csv

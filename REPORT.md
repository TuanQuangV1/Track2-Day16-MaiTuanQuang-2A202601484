# Báo cáo Thực hành LAB 16: Cloud AI Environment Setup (Microsoft Azure)

**Họ và tên:** Mai Tuấn Quang  
**Mã sinh viên / ID:** 2A202601484  
**Cloud chọn thực hành:** Microsoft Azure  
**Region:** `eastus`  
**Resource Group:** `ai-lab-rg`  
**Cấu hình máy ảo:** CPU Node (`Standard_B2s` - 2 vCPU, 4GB RAM, Ubuntu 22.04 LTS)  

---

## 1. Tổng quan Hạ tầng & Cấu hình (Checkpoints 0 - 3)
- **IAM & Security:** Sử dụng tài khoản Azure cá nhân giới hạn quyền trong Resource Group `ai-lab-rg`. Đã tạo Network Security Group (`ai-lab-nsg`) giới hạn truy cập SSH (Port 22) chỉ cho địa chỉ IP Public của cá nhân (`116.97.69.102/32`).
- **Bootstrap Environment:** Sử dụng `cloud-init-cpu.yaml` tự động cài đặt `python3-pip`, `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `kaggle` ngay khi khởi tạo VM.
- **Xác minh kết nối:** Đã kết nối SSH an toàn vào VM (`tuanq@52.224.241.220`) và kiểm tra `import lightgbm, sklearn, pandas, numpy` trả về `OK`.
- **Minh chứng:** [Screenshot/cp3_python_bootstrap_ok.png](file:///c:/Users/tuanq/Downloads/VinAI/Phase%202/Track2-Day16-MaiTuanQuang-2A202601484/Screenshot/cp3_python_bootstrap_ok.png)

---

## 2. Kết quả Benchmark Mô hình LightGBM (Checkpoint 4)
- **Bộ dữ liệu:** Credit Card Fraud Detection (284,807 dòng giao dịch).
- **Mô hình:** LightGBM Classifier (100 estimators, learning_rate=0.1, random_state=42).
- **File kết quả chi tiết:** `benchmark_result.json`
- **Minh chứng:** [Screenshot/cp4_lightgbm_benchmark.png](file:///c:/Users/tuanq/Downloads/VinAI/Phase%202/Track2-Day16-MaiTuanQuang-2A202601484/Screenshot/cp4_lightgbm_benchmark.png)

### Bảng tổng hợp Metrics thu được:

| Chỉ số (Metric) | Kết quả thực tế |
|---|---|
| Thời gian load data (`data_loading_time_seconds`) | **1.7508 giây** |
| Thời gian huấn luyện (`training_time_seconds`) | **2.2780 giây** |
| Số vòng lặp tối ưu (`best_iteration`) | **100** |
| **AUC-ROC** | **0.806111** |
| **Accuracy** | **0.998455 (99.85%)** |
| **F1-Score** | **0.584906** |
| **Precision** | **0.543860** |
| **Recall** | **0.632653** |
| Độ trễ dự đoán 1 mẫu (`inference_latency_ms_per_row`) | **0.8780 ms** |
| Thông lượng dự đoán (`inference_throughput_rows_per_sec`) | **299,413.49 mẫu/giây** |

---

## 3. Nhận xét & Đánh giá Chuyên sâu (Checkpoint 5)

### 3.1. Đánh giá Hiệu năng & Tối ưu hóa
- **Thời gian xử lý:** Việc huấn luyện mô hình LightGBM trên bộ dữ liệu gần 285k dòng chỉ mất hơn 2 giây trên VM CPU `Standard_B2s`. 
- **Độ trễ và Thông lượng:** Độ trễ dự đoán 1 mẫu đơn lẻ chỉ mất ~0.88ms, đồng thời thông lượng xử lý hàng loạt đạt xấp xỉ 300,000 giao dịch/giây. Điều này chứng minh CPU instance quy mô nhỏ hoàn toàn đáp ứng tốt cho các bài toán Real-time Fraud Detection ở quy mô thực tế mà chưa cần tốn chi phí cho GPU.

### 3.2. Vì sao Accuracy không đủ để đánh giá bài toán Fraud Detection?
- Dataset **Credit Card Fraud Detection** là bộ dữ liệu có đặc tính **mất cân bằng lớp cực kỳ nghiêm trọng (Imbalanced Dataset)**: trong 284,807 giao dịch thì chỉ có 492 giao dịch gian lận (~0.17%), còn lại 99.83% là giao dịch hợp lệ.
- Nếu xây dựng một mô hình "ngây thơ" chỉ đơn giản luôn phán đoán *mọi giao dịch đều hợp lệ*, mô hình đó vẫn dễ dàng đạt chỉ số **Accuracy = 99.83%**. Tuy nhiên, mô hình đó hoàn toàn vô dụng trong thực tế vì bỏ sót 100% gian lận (Recall = 0, F1-Score = 0).
- Do đó, để đánh giá đúng chất lượng phát hiện gian lận, ta bắt buộc phải dựa vào các chỉ số **AUC-ROC** (`0.8061`), **F1-Score** (`0.5849`), **Precision** (`0.5439`) và **Recall** (`0.6327`).

### 3.3. Giám sát Tài nguyên & Quản lý Chi phí
- **Tài nguyên VM:** Trong quá trình benchmark, RAM tiêu thụ chỉ khoảng 245MB / 3.8GB, CPU usage quay trở lại mức 0% sau khi hoàn thành. (Minh chứng: [Screenshot/cp5_vm_resource_monitoring.png](file:///c:/Users/tuanq/Downloads/VinAI/Phase%202/Track2-Day16-MaiTuanQuang-2A202601484/Screenshot/cp5_vm_resource_monitoring.png)).
- **Chi phí (Billing):** Tổng chi phí thực tế ghi nhận trên Azure Cost Management cho toàn bộ bài lab là **< $0.01** (Minh chứng: [Screenshot/cp5_azure_cost_analysis.png](file:///c:/Users/tuanq/Downloads/VinAI/Phase%202/Track2-Day16-MaiTuanQuang-2A202601484/Screenshot/cp5_azure_cost_analysis.png)).

---

## 4. Dọn dẹp Tài nguyên (Checkpoint 6)
- Đã thực hiện lệnh xóa toàn bộ Resource Group `ai-lab-rg` để đảm bảo không phát sinh chi phí duy trì:
  ```bash
  az group delete --name ai-lab-rg --yes --no-wait
  ```
- Đã xác minh trên Azure Console / CLI rằng toàn bộ tài nguyên (VM, VNet, Public IP, NSG) đã được hủy hoàn toàn.

---
**Xác nhận:** Bài làm hoàn thành đầy đủ 7 Checkpoints theo đúng tiêu chuẩn yêu cầu.

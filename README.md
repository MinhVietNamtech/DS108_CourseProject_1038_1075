<div align="center">

# 📊 DS108 – F&B Data Collection and Preprocessing

### Thu thập dữ liệu F&B và tiền xử lý cho bài toán phân loại mức độ yêu thích

**Môn học:** Tiền xử lý và xây dựng bộ dữ liệu (DS108)

**Giảng viên hướng dẫn:** TS. Nguyễn Gia Tuấn Anh, CN. Trần Quốc Khánh

| Thành viên            | MSSV     |
| --------------------- | -------- |
| **Võ Thụy Sao Mai**   | 24521038 |
| **Nguyễn Quang Minh** | 24521075 |

</div>

---

# 📖 Tổng quan dự án

Dự án xây dựng một bộ dữ liệu F&B (Food & Beverage) từ hai nền tảng **ShopeeFood** và **Foody**, sau đó thực hiện quá trình tiền xử lý nhằm tạo ra tập dữ liệu có cấu trúc phục vụ bài toán:

> **Phân loại mức độ yêu thích của khách hàng đối với quán ăn/nhà hàng (Low – Moderate – High).**

Bộ dữ liệu bao gồm:

* Thông tin quán ăn/nhà hàng
* Thông tin đánh giá và bình luận khách hàng
* Điểm cảm xúc được trích xuất bằng PhoBERT
* Các đặc trưng vận hành của quán
* Nhãn mục tiêu (Target Label)

Dữ liệu được thu thập tại 6 tỉnh/thành thuộc khu vực miền Nam Việt Nam.

---

# 🎯 Mục tiêu

Mục tiêu chính của dự án:

* Thu thập dữ liệu từ ShopeeFood và Foody.
* Làm sạch và chuẩn hóa dữ liệu.
* Xử lý dữ liệu thiếu và dữ liệu ngoại lai.
* Trích xuất cảm xúc từ bình luận khách hàng.
* Xây dựng nhãn phản ánh mức độ yêu thích của khách hàng.
* Tạo tập dữ liệu sẵn sàng cho các mô hình Machine Learning.

---

# 📂 Cấu trúc thư mục

```text
DS108_CourseProject_1038_1075/
├── data
│   ├── data_processed
│   │   ├── test
│   │   └── train
│   └── data_raw
│       ├── foody_csv
│       ├── shopee_csv
│       ├── sp_fd_csv
│       ├── test
│       ├── train
│       └── txt
├── notebooks
│   ├── instruction_list.ipynb
│   ├── n1_data_collection
│   │   ├── crawl_from_foody.ipynb
│   │   ├── crawl_from_shopeefood.ipynb
│   │   ├── foody_opt_v14.py
│   │   └── shopee_tools.py
│   ├── n2_data_cleaning_and_imputation
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── imputation.py
│   │   └── utils.py
│   └── n3_exploratory_data_analysis
│       ├── area_location_mapping.csv
│       ├── area_location_mapping_debug.csv
│       ├── define_address_v2.ipynb
│       ├── EDA_preprocessed.ipynb
│       ├── EDA_raw.ipynb
│       └── utils.py
├── README.md
└── requirements.txt
```

---

# 🔄 Pipeline xử lý dữ liệu

## 1. Thu thập dữ liệu

Nguồn dữ liệu:

* ShopeeFood
* Foody

Dữ liệu được chia thành:

### D Files – Thông tin quán ăn/nhà hàng

Mỗi bản ghi đại diện cho một quán ăn hoặc nhà hàng, bao gồm các thuộc tính:

| Thuộc tính | Mô tả |
|------------|--------|
| `RestaurantID` | Mã định danh quán |
| `Restaurant Name` | Tên quán |
| `Latitude` | Vĩ độ |
| `Longitude` | Kinh độ |
| `Address` | Địa chỉ |
| `Type` | Loại hình kinh doanh (quán ăn, nhà hàng, cafe, ...) |
| `Cuisines` | Các món ăn hoặc nhóm món ăn phục vụ |
| `Monday` → `Sunday` | Thời gian hoạt động theo từng ngày trong tuần |
| `Style` | Phong cách ẩm thực |
| `Has_Online_delivery` | Có hỗ trợ giao hàng trực tuyến |
| `Has_Table_booking` | Có hỗ trợ đặt bàn |
| `Has_Wifi` | Có cung cấp Wifi |
| `Price` | Khoảng giá |
| `Product_quality(10)` | Điểm chất lượng món ăn (thang 10) |
| `Serving_quality(10)` | Điểm chất lượng phục vụ (thang 10) |
| `Interior_design(10)` | Điểm đánh giá không gian/quán (thang 10) |
| `Average_rating(5)` | Điểm đánh giá trung bình (thang 5) |
| `Total votes` | Tổng số lượt đánh giá |

```
```
### C Files – Dữ liệu bình luận khách hàng

Mỗi bản ghi đại diện cho một lượt đánh giá của người dùng đối với một quán ăn/nhà hàng, bao gồm các thuộc tính:

| Thuộc tính | Mô tả |
|------------|--------|
| `UserID` | Mã định danh người dùng |
| `User` | Tên hoặc biệt danh người dùng |
| `Review Time` | Thời điểm đăng đánh giá |
| `Rating (10)` | Điểm đánh giá của người dùng (thang 10) |
| `Comment` | Nội dung bình luận |
| `RestaurantID` | Mã quán được đánh giá (khóa ngoại liên kết với D Files) |

```
```
---

## 2. Phân tích dữ liệu thô

Thực hiện thống kê:

- Số lượng bản ghi
- Giá trị thiếu (Missing Values)
- Dữ liệu trùng lặp
- Tệp dữ liệu rỗng
- Dữ liệu không liên kết được với địa điểm/quán ăn
- Phân bố độ dài bình luận
- Mất cân bằng giữa các nguồn dữ liệu

---

## 3. Trích xuất cảm xúc

Bình luận được xử lý bằng:

* PhoBERT

Quy trình:

```text
Comment
   ↓
Làm sạch bình luận
   ↓
Token hóa
   ↓
PhoBERT
   ↓
Tỉ lệ tích cực
   ↓
Điểm Sentiment Score
```

Điểm cảm xúc được quy đổi về thang điểm 5.

---

## 4. Xây dựng biến mục tiêu

Điểm yêu thích cuối cùng được tạo bằng cách kết hợp:

* Rating của nền tảng
* Sentiment Score từ PhoBERT

Sau đó phân lớp thành:

| Nhãn     | Ý nghĩa           |
| -------- | ----------------- |
| Low      | Ít được yêu thích |
| Moderate | Trung bình        |
| High     | Được yêu thích    |

---

## 5. Gộp dữ liệu

Entity Resolution giữa ShopeeFood và Foody dựa trên các thuộc tính:

* Restaurant Name
* Address
* Latitude
* Longitude

Thực hiện:

- Đối sánh lược đồ dữ liệu (Schema Matching)
- Đối chiếu thực thể (Entity Matching)
- Loại bỏ dữ liệu trùng lặp (Deduplication)
- Chuẩn hóa thực thể (Entity Normalization)

---

## 6. Chia Train/Test

```python
train_test_split(
    test_size=0.2,
    random_state=42
)
```

Tỷ lệ:

```text
Train : 80%
Test  : 20%
```

Việc chia dữ liệu được thực hiện trước toàn bộ bước xử lý học thống kê nhằm tránh Data Leakage.

---

## 7. Làm sạch dữ liệu

### Loại bỏ

* Quán không thuộc lĩnh vực F&B
* Dữ liệu lỗi
* Bản ghi trùng lặp

### Chuẩn hóa

- Tên quán (`Restaurant Name`)
- Địa chỉ (`Address`)
- Loại hình kinh doanh (`Type`)
- Phong cách ẩm thực (`Style`)

---

## 8. Xử lý ngoại lai

Áp dụng phương pháp:

### Interquartile Range (IQR)

```text
IQR = Q3 - Q1
```

Ngưỡng:

```text
Lower = Q1 - 1.5 × IQR
Upper = Q3 + 1.5 × IQR
```

Áp dụng cho:

* Min_Price
* Max_Price

---

## 9. Xử lý dữ liệu thiếu

### Style

Suy luận từ:

* Cuisines

### Cuisines

Suy luận từ:

* Tên quán
* Style

### Type

Suy luận từ:

* Cuisines
* Price Bucket

### Price

Suy luận theo:

```text
(City, District, Cuisines)
```

### Monday - Sunday

Trích xuất thành:

* Avg_open_hour
* Avg_close_hour
* Avg_open_duration
* Is_open_weekend
* Is_open_after_22

---

## 10. Feature Engineering

Các đặc trưng được xây dựng gồm:

### Giá

* Min_Price
* Max_Price
* Avg_Price
* Price_Range

### Thời gian

* Avg_open_hour
* Avg_close_hour
* Avg_open_duration
* Is_open_weekend
* Is_open_after_22

### Địa lý

* City
* District

### Ẩm thực

* Style
* Type
* Cuisines

### Đánh giá

* Shopee_points(5)
* Foody_points(5)

---

## 11. Encoding

Các biến phân loại được chuyển thành dạng số bằng:

* One-Hot Encoding
* Multi-Hot Encoding

---

## 12. Xuất dữ liệu cuối cùng

Kết quả:

```text
train_unscaled.csv
test_unscaled.csv
```

sẵn sàng cho các mô hình Machine Learning.

---

# 🚀 Reproduce

## 1. Clone repository

```bash
git clone https://github.com/MinhVietNamtech/DS108_CourseProject_1038_1075.git
cd DS108_CourseProject_1038_1075
```

## 2. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

## 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 4. Hướng dẫn vận hành

Đã có hướng dẫn chi tiết cách chạy code trong file **instruction_list.ipynb**

---
# 📈 Kết quả

Bộ dữ liệu đầu ra:

* Đã loại bỏ dữ liệu nhiễu.
* Đã xử lý dữ liệu thiếu.
* Đã chuẩn hóa đặc trưng.
* Có nhãn Low / Moderate / High.
* Sẵn sàng cho các bài toán Machine Learning.

---

# 📜 License

Dự án phục vụ mục đích học tập trong môn DS108 – Trường Đại học Công nghệ Thông tin (UIT – VNUHCM).

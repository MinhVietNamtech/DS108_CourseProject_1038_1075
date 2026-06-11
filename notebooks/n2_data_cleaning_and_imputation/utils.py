import numpy as np
import pandas as pd
import re
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from n2_data_cleaning_and_imputation.feature_engineering import duplicate_emoji_handling, calculate_foody_points
from n2_data_cleaning_and_imputation.cleaning import normalize_districts
import math
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Hàm xóa cột
def drop_selected_columns(df, cols_to_drop):
    df = df.copy()
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols, errors="ignore")
    return df

# Hàm tính tần suất xuất hiện của từng phần tử cho các cột đa nhãn
def calculate_style_frequencies(df, column_name='Style'):
    # 1. Loại bỏ các dòng bị khuyết (NaN) ở cột Style để tránh lỗi
    style_column = df[column_name].dropna()
    
    # 2. Tách chuỗi theo dấu phẩy -> Chuyển mỗi dòng thành một List các style con
    style_lists = style_column.str.split(',')
    
    # 3. Dùng explode() để "nổ" các list thành từng dòng độc lập
    style_exploded = style_lists.explode()
    
    # 4. Xóa bỏ khoảng trắng thừa ở đầu và cuối của từng chữ sau khi tách
    style_cleaned = style_exploded.str.strip()
    
    # 5. Loại bỏ các chuỗi rỗng (nếu có dòng bị dư dấu phẩy ở cuối như "Món Việt, ")
    style_cleaned = style_cleaned[style_cleaned != ""]
    
    # 6. Đếm tần số xuất hiện của từng phần tử độc nhất
    style_counts = style_cleaned.value_counts()
    
    return style_counts

# Hàm kết hợp dữ liệu từ file shopee và foody (nếu có) và tạo thêm cột cần thiết
def combine_shopee_foody(shopee_idx: int = None, foody_idx: int = None):
    '''
    Kết hợp dữ liệu từ foody và shopee sau khi đã tính điểm bình luận
    Tạo thêm cột Foody_points(5) và Shopee_points(5)
    '''
    foody = None
    shopee = None

    # Xử lý file foody (nếu có)
    if foody_idx is not None:
        print(f'Có file foody tăng cường! ID: {foody_idx}')
        try:
            # Thử đọc file
            foody_C = pd.read_csv(f"../data/data_raw/foody_csv/C/C{foody_idx}.csv")
        except pd.errors.EmptyDataError:
            # Nếu file rỗng, gán thành DataFrame trống
            print(f"Cảnh báo: File C{foody_idx}.csv rỗng. Đang bỏ qua...")
            foody_C = pd.DataFrame()
        foody_C = duplicate_emoji_handling(foody_C)
        
        foody = pd.read_csv(f"../data/data_raw/foody_csv/D/D{foody_idx}.csv")
        foody['RestaurantID'] = foody['RestaurantID'].astype(str)
        
        foody_point = calculate_foody_points(foody_C)
        foody_point['RestaurantID'] = foody_point['RestaurantID'].astype(str)

        # Bỏ dòng không có đánh giá
        foody = foody[(foody['Average_rating(5)'] != "_._") & foody['Average_rating(5)'].notna()].copy()
        foody = pd.merge(foody, foody_point[['RestaurantID', 'Foody_points(5)']], on='RestaurantID', how='left')
        foody['Average_rating(5)'] = pd.to_numeric(foody['Average_rating(5)'], errors='coerce')
        foody['Foody_points(5)'] = foody['Foody_points(5)'].fillna(foody['Average_rating(5)'])
        
        # Bỏ cột cũ và đồng bộ cột Shopee_points gán bằng 0.0
        foody = foody.drop(columns=['Average_rating(5)', 'Total votes'], errors='ignore')
        foody['Shopee_points(5)'] = 0.0 
    
    # Xử lý file shopee (nếu có)
    if shopee_idx is not None:  # Sửa lỗi kiểm tra nếu truyền idx = 0
        print(f'Có file shopee! ID: {shopee_idx}')
        shopee = pd.read_csv(f"../data/data_raw/shopee_csv/D/D{shopee_idx}.csv")

        # Bỏ dòng không có đánh giá
        shopee = shopee[shopee['Total votes'].notna()].copy()
        shopee['Shopee_points(5)'] = shopee['Average_rating(5)']
        shopee = shopee.drop(columns=['Average_rating(5)', 'Total votes'], errors='ignore')

        shopee_C = pd.read_csv(f"../data/data_raw/shopee_csv/C/C{shopee_idx}.csv")
        if len(shopee_C) != 0:
            shopee_C = duplicate_emoji_handling(shopee_C)
            shopee_points = calculate_foody_points(shopee_C)
            shopee_points['RestaurantID'] = shopee_points['RestaurantID'].astype(str)
            shopee['RestaurantID'] = shopee['RestaurantID'].astype(str)
            
            shopee = pd.merge(shopee, shopee_points[['RestaurantID', 'Foody_points(5)']], on='RestaurantID', how='left')
            shopee['Foody_points(5)'] = shopee['Foody_points(5)'].fillna(0.0)
        else:
            shopee['Foody_points(5)'] = 0.0

    # Kết hợp shopee và foody
    if shopee is not None and foody is None:
        return shopee
    elif foody is not None and shopee is None:
        return foody
    elif shopee is not None and foody is not None:
        shopee['Restaurant Name'] = shopee['Restaurant Name'].astype(str).str.strip()
        foody['Restaurant Name'] = foody['Restaurant Name'].astype(str).str.strip()

        # 1. Làm tròn tọa độ để đảm bảo khớp dữ liệu (sai số nhỏ không ảnh hưởng)
        # 5 chữ số thập phân tương đương với độ chính xác khoảng 1 mét
        shopee['Lat_Round'] = shopee['Latitude'].round(5)
        shopee['Long_Round'] = shopee['Longitude'].round(5)
        foody['Lat_Round'] = foody['Latitude'].round(5)
        foody['Long_Round'] = foody['Longitude'].round(5)

        # 2. Tìm quán trùng dựa trên ID và Tọa độ đã làm tròn
        dup_restaurant = pd.merge(
            shopee[['Restaurant Name', 'Address', 'Lat_Round', 'Long_Round']], 
            foody[['Restaurant Name', 'Address', 'Lat_Round', 'Long_Round']], 
            on=['Restaurant Name', 'Address', 'Lat_Round', 'Long_Round'], 
            how='inner'
        )
        
        # Lấy danh sách trùng
        dup_id = dup_restaurant['Restaurant Name'].unique()
        
        # 3. Bỏ quán trùng trong file foody vì dữ liệu trong shopee đã bao gồm điểm cả 2 nguồn
        foody = foody[~foody['Restaurant Name'].isin(dup_id)].copy()
        
        # Xóa các cột phụ đã tạo
        shopee = shopee.drop(columns=['Lat_Round', 'Long_Round'])
        foody = foody.drop(columns=['Lat_Round', 'Long_Round'])
        
        # 4. Đảm bảo thứ tự cột và nối dữ liệu
        foody = foody.reindex(columns=shopee.columns)
        sp_fd = pd.concat([shopee, foody], axis=0, ignore_index=True)
        return sp_fd
        
    return None

# Hàm gán nhãn
def label_annotating(row, q1, q2):
    """
    Tính điểm trung bình từ Shopee + Foody,
    sau đó gán nhãn theo quantile 1/3 và 2/3:
      score <= q1           -> 'low'
      q1 < score <= q2      -> 'moderate'
      score > q2            -> 'high'
    """
    if row['Shopee_points(5)']==np.nan and row['Foody_points(5)']==np.nan:
        return 'unknown'
    if row['Shopee_points(5)'] == 0.0:
        score = row['Foody_points(5)']
    elif row['Foody_points(5)'] == 0.0:
        score = row['Shopee_points(5)']
    else:
        score = (row['Shopee_points(5)'] + row['Foody_points(5)']) / 2

    if score <= q1:
        return 'low'
    elif score <= q2:
        return 'moderate'
    else:
        return 'high'

# Lớp gán nhãn
class LabelProcessor:
    def __init__(self, label_annotating_func):
        self.label_annotating_func = label_annotating_func
        self.q1 = None
        self.q2 = None

    def _calculate_avg_score(self, df):
        return df.apply(
            lambda row: row['Foody_points(5)'] if row['Shopee_points(5)'] == 0.0
            else row['Shopee_points(5)'] if row['Foody_points(5)'] == 0.0
            else (row['Shopee_points(5)'] + row['Foody_points(5)']) / 2,
            axis=1
        )

    def fit(self, df):
        """Tính toán ngưỡng q1, q2 từ tập train."""
        avg_scores = self._calculate_avg_score(df)
        self.q1 = avg_scores.quantile(1/3)
        self.q2 = avg_scores.quantile(2/3)
        print(f"Đã fit xong: q1={self.q1:.4f}, q2={self.q2:.4f}")
        return self

    def transform(self, df):
        """Áp dụng gán nhãn dựa trên ngưỡng đã fit."""
        if self.q1 is None or self.q2 is None:
            raise ValueError("Bạn phải gọi hàm fit() trước khi gọi transform().")
            
        df_out = df.copy()
        
        # Gán nhãn
        df_out['preference_level'] = df_out.apply(
            lambda row: self.label_annotating_func(row, self.q1, self.q2), axis=1
        )
        
        # Làm sạch dữ liệu
        df_out = df_out.dropna(subset=['preference_level']).copy()
        df_out['preference_level'] = df_out['preference_level'].astype(str)
        df_out = df_out.drop(columns=['Shopee_points(5)', 'Foody_points(5)'], errors='ignore')
        
        # Đánh lại ID
        df_out["RestaurantID"] = np.arange(1, len(df_out) + 1)
        
        return df_out

    def fit_transform(self, df):
        """Kết hợp fit và transform."""
        return self.fit(df).transform(df)

# Hàm đọc quota từ Excel và trả về dict {(city_lower, district_lower): n_keep}
def load_quota_from_excel(excel_path, sheet_name='Giảm mẫu',
                          city_col='Tỉnh', district_col='Quận', quota_col='Scaled'):
    """
    Mục đích: kiểm tra tỉ lệ quán để quyết định có cần thu thập thêm không
    Đọc bảng quota từ sheet 'Giảm mẫu', trả về dict {(city_lower, district_lower): n_keep}.

    Cấu trúc file:
        - Cột Tỉnh: pandas đã ffill sẵn (không có ô merge thật)
        - 2 hàng cuối là 'Tổng' và 'Số mẫu' → Scaled = NaN → tự bị loại
        - Scaled là float -> cast sang int
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Bỏ hàng không có Quận hoặc Scaled (Tổng, Số mẫu, hàng trống)
    df = df[df[district_col].notna() & df[quota_col].notna()].copy()
    df = df[df[district_col].astype(str).str.strip() != '']
    # Bỏ đơn vị hành chính cho quận
    df = normalize_districts(df, district_col=district_col)

    # Chuẩn hoá lower + strip để match với df dữ liệu
    # df['_city_key']     = df[city_col].astype(str).str.strip().str.lower()
    df['_city_key'] = (
        df[city_col].astype(str)
            .str.strip()
            .str.lower()
            .str.replace('.', '', regex=False)   # bỏ dấu chấm
            .str.split()                          # tách theo khoảng trắng
            .str.join('')                        # nối lại 1 space (loại double space)
        )
    df['_district_key'] = df[district_col].astype(str).str.strip().str.lower()
    df['_quota']        = df[quota_col].astype(int)
    quota = dict(zip(
        zip(df['_city_key'], df['_district_key']),
        df['_quota']
    ))
    return quota

# Hàm lấy mẫu theo tỉ lệ trong file excel
def sample_by_district(df, quota, city_col='City', district_col='District',
                        random_state=42):
    """
    Giữ lại số hàng theo quota cho từng cặp (City, District).

    Tham số:
        df           : DataFrame cần lọc
        quota        : dict {(city_lower, district_lower): n_keep}
                       — lấy từ load_quota_from_excel()
        city_col     : tên cột tỉnh trong df
        district_col : tên cột quận trong df
        random_state : seed để tái lập kết quả

    Trả về:
        df_sampled   : DataFrame đã lọc
        report       : DataFrame báo cáo so sánh quota vs thực tế
    """
    df = df.copy()
    df['_city_key'] = (
            df[city_col].astype(str)
            .str.strip()
            .str.lower()
            .str.replace('.', '', regex=False)   # bỏ dấu chấm
            .str.split()                          # tách theo khoảng trắng
            .str.join('')
        )
    df['_district_key'] = df[district_col].astype(str).str.strip().str.lower()

    sampled_parts = []
    report_rows   = []

    for (city_key, district_key), n_scaled in quota.items():
        n_keep = n_scaled
        if (n_keep < 20):
            n_keep = 20 # Đặt mức tối thiểu là 20 để đảm bảo đủ dữ liệu cho mô hình học máy
        # mask      = (df['_city_key'] == city_key) & (df['_district_key'] == district_key)
        mask      = df['_city_key'].str.contains(city_key, na=False, regex=False) & df['_district_key'].str.contains(district_key, na=False, regex=False)
        group     = df[mask]
        available = len(group)

        if available == 0:
            report_rows.append({
                'City': city_key, 'District': district_key,
                'Quota': n_scaled, 'Available': 0, 'Kept': 0,
                'Status': f'Không có dữ liệu về {city_key}, {district_key}'
            })
            continue

        if available < n_keep: # bé hơn thì lấy hết, nhưng báo lỗi
            sampled_parts.append(group)
            report_rows.append({
                'City': city_key, 'District': district_key,
                'Quota': n_scaled, 'Available': available, 'Kept': available,
                'Status': f'Thiếu {n_keep - available} hàng'
            })
        else:
            sampled_parts.append(group.sample(n=n_keep, random_state=random_state))
            report_rows.append({
                'City': city_key, 'District': district_key,
                'Quota': n_scaled, 'Available': available, 'Kept': n_keep,
                'Status': 'Đủ'
            })

    df_sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    df_sampled = df_sampled.drop(columns=['_city_key', '_district_key'])

    report = pd.DataFrame(report_rows)

    print(f"\n{'='*55}")
    print(f"Tổng cột Scaled   : {report['Quota'].sum():,}")
    print(f"Tổng giữ lại : {report['Kept'].sum():,}")
    print(f"Đủ quota     : {(report['Status'] == 'Đủ').sum()} quận")
    print(f"Không có dữ liệu: {report['Status'].str.startswith('Không có dữ liệu').sum()} quận")
    print(f"Thiếu dữ liệu: {report['Status'].str.startswith('Thiếu').sum()} quận")
    print(f"{'='*55}")

    return df_sampled, report

# Hàm tính tương quan và đề xuất xóa
def analyze_and_filter_correlations(df, target_col, threshold=0.85):
    """
    Phân tích tương quan của tất cả cột (số và nhị phân) so với target
    và tìm các cột có tương quan quá cao với nhau (cần loại bỏ).
    """
    # 1. Tính ma trận tương quan toàn bộ
    corr_matrix = df.corr()
    
    # 2. Lọc các cột có tương quan quá cao với nhau (để loại bỏ nhiễu)
    # Tạo ma trận tam giác trên để chỉ xét mỗi cặp 1 lần
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Tìm các cột có tương quan > threshold
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    
    # 3. Phân tích độ tương quan với biến mục tiêu (target)
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    
    print(f"--- Đề xuất loại bỏ do tương quan quá cao (> {threshold}): ---")
    print(to_drop)
    
    print(f"\n--- Độ tương quan với '{target_col}': ---")
    print(target_corr.drop(target_col))

# Hàm tách ô đa nhãn
def split_multilabel_cell(value):
    """
    Tách ô multi-label dạng:
    'Quán ăn, Cafe'
    thành:
    ['Quán ăn', 'Cafe']
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none"]:
        return []

    return [item.strip() for item in text.split(",") if item.strip()]

# Hàm vẽ biểu đồ các cột cần scaling
def plot_need_scaling(combined_df, need_scaling, cols_per_row=3):
    n_cols = len(need_scaling)
    n_rows = math.ceil(n_cols / cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 4, n_rows * 3))
    axes = axes.flatten()
    
    for i, col in enumerate(need_scaling):
        axes[i].hist(combined_df[col].dropna(), bins=20, color='skyblue', edgecolor='white')
        axes[i].set_title(f'Phân bổ: {col}', fontsize=10)
        axes[i].tick_params(axis='x', which='both', bottom=False, labelbottom=False) # Ẩn nhãn trục x
        
    # Ẩn các khung hình thừa nếu số lượng biểu đồ không chia hết cho số ô
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

# Hàm phân tích tương quan và đề xuất bỏ cột
def analyze_and_filter_correlations(df, target_col, threshold=0.85):
    """
    Phân tích tương quan của tất cả cột (số và nhị phân) so với target
    và tìm các cột có tương quan quá cao với nhau (cần loại bỏ).
    """
    # 1. Tính ma trận tương quan toàn bộ
    corr_matrix = df.corr()
    
    # 2. Lọc các cột có tương quan quá cao với nhau (để loại bỏ nhiễu)
    # Tạo ma trận tam giác trên để chỉ xét mỗi cặp 1 lần
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Tìm các cột có tương quan > threshold
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    
    # 3. Phân tích độ tương quan với biến mục tiêu (target)
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    
    print(f"--- Đề xuất loại bỏ do tương quan quá cao (> {threshold}): ---")
    print(to_drop)
    
    print(f"\n--- Độ tương quan với '{target_col}': ---")
    print(target_corr.drop(target_col))
    
    return to_drop, target_corr
import numpy as np
import pandas as pd
import re
from transformers import pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter, defaultdict
from itertools import combinations
from n2_data_cleaning_and_imputation.cleaning import normalize_districts

# Từ điển các món ăn
FOOD_CATEGORY_MAPPING = {
    'Cuisines_MainDish': [
        'bún', 'cơm', 'mì', 'món nước', 'hủ tiếu', 'cơm tấm', 'phở', 'miến',
        'lẩu', 'cơm văn phòng', 'bánh mì', 'cơm chiên', 'cháo', 'bánh canh',
        'bánh đa cua', 'cơm tay cầm', 'bò kho', 'món huế', 'phở cuốn',
        'beefsteak - bò né', 'sườn nướng', 'thịt bò', 'thịt gà', 'thịt heo',
        'cá', 'tôm', 'mực', 'hải sản', 'nghêu - sò - ốc', 'thịt vịt', 'gà ta',  
        'heo quay', 'gà xối mỡ', 'thịt ếch', 'gà ác/gà hầm', 'vịt quay',
        'bồ câu', 'bò lá lốt', 'vi cá', 'bào ngư', 'nấm', 'đậu hũ', 'dimsum', 'hamburger',
        'bánh bao', 'xôi', 'bánh ướt', 'bánh bèo', 'bánh xèo'
    ],
    'Cuisines_Beverage': [
        'thức uống', 'sinh tố - nước ép', 'trà sữa', 'trà chanh', 'trà', 'nước', 'cafe', 'kem',
        'sữa', 'café - kem', 'nước ngọt', 'bia', 'cocktail', 'rượu vang', 'rượu'
    ],
    'Cuisines_Dessert': [
        'bánh - kẹo', 'bánh', 'kẹo', 'chè', 'trái cây', 'bánh sinh nhật/bánh kem',
        'donut', 'cupcake', 'cake', 'bánh su/chour'
    ],
    'Cuisines_Snack': [
        'ăn vặt - ăn nhẹ', 'fastfood - thức ăn nhanh', 'gà rán', 'sushi & sashimi',
        'nem', 'xúc xích', 'salad', 'cá/bò viên',
        'bánh tằm', 'chân cánh gà nướng', 'hoành thánh - vằn thắn',
        'sủi cảo', 'cua - ghẹ', 'gỏi', 'bánh tráng',
        'súp', 'bò nướng', 'chả giò', 'cơm cháy', 'bánh cuốn',
        'pho mai que', 'bánh căn', 'phá lấu', 'bánh hỏi', 'hàu', 'bánh đúc',
        'bánh khọt', 'trứng vịt lộn'
    ]
}
# List các cột nhị phân thể hiện tiện ích quán
BINARY_COLS = [
    "Has_Online_delivery",
    "Has_Table_booking",
    "Has_Wifi"
]
# List các cột thời gian mở cửa
DAY_COLS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]
# Hàm tính điểm từ bình luận foody
def calculate_foody_points(comments_df: pd.DataFrame):
    if comments_df.empty:
        return pd.DataFrame(columns=['RestaurantID', 'Foody_points(5)'])
    # Load model Pho-BERT
    classifier = pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment", top_k=None)

    def sentiment_anal(x):
        total_point = 0.0
        # Chỉ đếm những bình luận có chữ để tính trung bình
        valid_comments = x[x['Comment'].notna() & (x['Comment'].str.strip() != "")]
        num_valid = len(valid_comments)
        
        if num_valid == 0:
            return pd.Series({'Foody_points(5)': 0.0})

        for _, row in valid_comments.iterrows():
            try: # Lấy điểm đánh giá từ người dùng trên hệ thống
                rating = float(row['Rating (10)'])
            except (ValueError, TypeError):
                rating = np.nan
            
            try:
                comment_text = str(row['Comment'])
                predicts = classifier(comment_text, truncation=True, max_length=256)
                
                # Tìm điểm POS trong list kết quả
                pos_score = next((p['score'] for p in predicts[0] if p['label'] == 'POS'), 0.5)
                post_point = np.round(pos_score, 1) * 10.0
            except Exception as e:
                post_point = 5.0 # Nếu model lỗi, lấy điểm trung bình (5/10)
            
            # Tính toán total_point
            if np.isnan(rating):
                total_point += post_point
            else:
                total_point += (rating + post_point) / 2
                
        avg_point = total_point / num_valid
        return pd.Series({'Foody_points(5)': np.round((avg_point / 10) * 5, 1)})

    # Làm sạch dữ liệu trước khi groupby
    comments_df['Rating (10)'] = pd.to_numeric(comments_df['Rating (10)'], errors='coerce')
    
    # Dùng groupby
    res = comments_df.groupby('RestaurantID', as_index=False).apply(sentiment_anal)
    return res

# Hàm xử lý bình luận bị lặp và dịch emoji sang văn bản
def duplicate_emoji_handling(comments):
    n_raw = len(comments)
    if n_raw == 0:
        return comments
    idx=0
    drop_lst = []
    while idx < (n_raw-1):
        cur_idx = idx+1 # Next element
        while (comments['UserID'][cur_idx] == comments['UserID'][idx]) or type(comments['User'][cur_idx])!=str:
            drop_lst.append(cur_idx)
            cur_idx += 1
            if cur_idx >= n_raw: 
                break
        idx = cur_idx
    unique_com = comments.drop(drop_lst) # Bỏ dòng lặp

    # Dịch emoji sang văn bản
    trans_map = {
        ':))': 'vui vẻ', '^^': 'vui vẻ', ')))': 'vui vẻ', '@@': 'khóc thút thít',
        '❤': 'yêu thích', '🍜': 'ngon', '🥤': 'nước', '☑': 'yêu thích', '✔': 'yêu thích',
        '🔜': 'sớm', '🥬': 'rau xanh', '♥': 'yêu thích', '🤤': 'thèm thuồng', '😋': 'thèm thuồng',
        '👍': 'yêu thích', '👏': 'tán thưởng', '😊': 'vui vẻ', '😚': 'khá yêu thích', '😘': 'yêu thích',
        '😍': 'cực yêu thích', '🙁': 'buồn bã', '⭕': 'duyệt', '🥰': 'yêu thương', '😃': 'vui vẻ',
        '🤔': 'nghi hoặc', '😂': 'cười sảng khoái', '🤣': 'cười cực sảng khoái', '🤩': 'phấn khởi', '😈': 'con quỷ'
    }
    unique_com['Comment'] = unique_com['Comment'].astype(str)
    for emo, meaning in trans_map.items():
        unique_com['Comment'] = unique_com['Comment'].str.replace(emo, f" {meaning} ", regex=False)

    return unique_com

# Hàm chuyển cột nhị phân từ yes/no -> 1/0
def encode_binary_columns(df, binary_cols=None):
    df = df.copy()

    if binary_cols is None:
        binary_cols = BINARY_COLS
    
    for col in binary_cols:
        if col in df.columns:
            cleaned = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            df[col] = cleaned.map({"yes": 1, "no": 0})

    return df

# Hàm trích xuất địa chỉ
def parse_address(combined_df):
    add_df = combined_df[["Address"]].copy()
    add_df['Address'] = add_df['Address'].fillna("___") # Thay "___" cho giá trị thiếu để dễ xử lý sau
    City = [row.split(', ')[-1] for row in add_df['Address']]
    District = [row.split(', ')[len(row.split(', ')) - 2] for row in add_df['Address']]
    Local_address = [(', '.join(row.split(', ')[:(len(row.split(', ')) - 2)])).strip() for row in add_df['Address']]
    combined_df['City'] = City
    combined_df['District'] = District
    combined_df['Local_address'] = Local_address

    # Chuẩn hóa tên quận/huyện
    combined_df = normalize_districts(combined_df)

    # Chuẩn hóa địa chỉ đầy đủ
    combined_df['Address'] = (
        combined_df[['Local_address', 'District', 'City']]
        .fillna('')
        .apply(lambda row: ', '.join(p for p in row if p.strip()).lower(), axis=1)
    )
    return combined_df

# Hàm tách Giá bán
def parse_price_value(value):
    """
    Parse chuỗi Price thành min_price và max_price.
    Ví dụ:
    '0 - 0' -> 0, 0
    '35.000 - 289.000' -> 35000, 289000
    '30.000đ - 50.000đ' -> 30000, 50000
    """
    if pd.isna(value):
        return pd.Series([float('nan'), float('nan')])

    text = str(value).strip()

    nums = re.findall(r"\d[\d\.]*", text)
    nums = [int(num.replace(".", "")) for num in nums]

    if len(nums) >= 2:
        return pd.Series([nums[0], nums[1]])

    if len(nums) == 1:
        return pd.Series([nums[0], nums[0]])

    return pd.Series([float('nan'), float('nan')])

# Hàm trích xuất các features Giá từ cột Price ban đầu
def create_price_features(df, price_col="Price", zero_as_missing=True):
    """
    Tạo các feature từ Price:
    - min_price
    - max_price
    - avg_price
    - price_range

    zero_as_missing=True:
    Nếu Price = 0 - 0 thì xem là thiếu giá, đổi thành NaN.
    Vì trong file mẫu, 0 - 0 nhiều khả năng là không có dữ liệu giá.
    """
    df = df.copy()

    if price_col not in df.columns:
        return df

    df[["Min_Price", "Max_Price"]] = df[price_col].apply(parse_price_value)

    df["Min_Price"] = pd.to_numeric(df["Min_Price"], errors="coerce")
    df["Max_Price"] = pd.to_numeric(df["Max_Price"], errors="coerce")

    if zero_as_missing:
        zero_price_mask = (df["Min_Price"] == 0) & (df["Max_Price"] == 0)
        df.loc[zero_price_mask, ["Min_Price", "Max_Price"]] = pd.NA

    df["Avg_Price"] = df[["Min_Price", "Max_Price"]].mean(axis=1)
    df["Price_range"] = df["Max_Price"] - df["Min_Price"]

    return df

# Hàm rút ra bộ món đặc trưng cho từng Style và điền Style/Cuisines cho các hàng thiếu dựa trên Jaccard similarity 
def extract_style_dish_map(df, target_col = 'Cuisines', cat_col = 'Style', dish_threshold=0.2, 
                           min_restaurant_count=2, dedup_jaccard=0.7, keep_merge = False):
    """
    Xây bộ món đặc trưng cho từng Style, gồm 3 bước:
      1. Thresholding  : chỉ giữ món xuất hiện >= dish_threshold * số quán của Style đó
      2. Style lọc yếu : loại Style có < min_restaurant_count quán (không đủ tin cậy)
      3. Deduplication : gộp các Style có bộ món gần giống nhau (Jaccard >= dedup_jaccard),
                         giữ lại Style phổ biến hơn
    """
    valid_df = df[df[cat_col].notna() & df[target_col].notna()].copy()
 
    style_counts        = Counter()
    dish_counts_per_style = defaultdict(Counter)
 
    # Bước 1: Đếm tần suất
    for _, row in valid_df.iterrows():
        styles = [s for s in re.split(r',\s*', str(row[cat_col])) if s.strip()]
        dishes = [d.lower() for d in re.split(r',\s*', str(row[target_col])) if d.strip()]
        for style in styles:
            style_counts[style] += 1
            for dish in dishes:
                dish_counts_per_style[style][dish] += 1
 
    # Bước 2: Thresholding + lọc Style yếu
    raw_style_map = {}   # {style: set(dishes)}
    raw_freq_map  = {}   # {style: {dish: freq}}  ← giữ lại tần suất
 
    for style, total in style_counts.items():
        if total < min_restaurant_count:
            continue
        min_dish_count = total * dish_threshold
        significant = {
            dish: count / total                          # tần suất [0.0–1.0]
            for dish, count in dish_counts_per_style[style].items()
            if count >= min_dish_count
        }
        if significant:
            raw_style_map[style] = set(significant.keys())
            raw_freq_map[style]  = significant
 
    # Bước 3: Deduplication bằng Jaccard
    merged = {}
    for s1, s2 in combinations(raw_style_map.keys(), 2):
        d1, d2 = raw_style_map[s1], raw_style_map[s2]
        jaccard = len(d1 & d2) / len(d1 | d2)
        if jaccard >= dedup_jaccard:
            keep   = s1 if style_counts[s1] >= style_counts[s2] else s2
            remove = s2 if keep == s1 else s1
            merged[remove] = keep
    if not keep_merge:
        style_dish_map = {s: d for s, d in raw_style_map.items() if s not in merged}
    else:
        style_dish_map = {s: d for s, d in raw_style_map.items()}
    dish_freq_map  = {s: f for s, f in raw_freq_map.items()}
 
    return style_dish_map, dish_freq_map, merged

# Hàm mã hóa cột Style
def encode_style(df, column_name='Style'):
    '''
    Mục tiêu: Phân loại các style con theo các quốc gia
    Kỹ thuật: Mã hóa Multi-hot encoding
    '''
    # 1. Từ điển ánh xạ 
    country_mapping = {
        # --- Vietnamese Cuisine ---
        'món việt': 'Vietnamese', 'món miền nam': 'Vietnamese', 'món bắc': 'Vietnamese',
        'món miền trung': 'Vietnamese', 'hà nội': 'Vietnamese', 'món huế': 'Vietnamese',
        'tây nguyên': 'Vietnamese', 'miền tây': 'Vietnamese', 'đặc sản vùng': 'Vietnamese',
        'tây bắc': 'Vietnamese', 'nam định': 'Vietnamese', 'đà lạt': 'Vietnamese',
        'miền đông': 'Vietnamese', 'món quảng': 'Vietnamese',
        
        # --- Asian Cuisine ---
        'món trung hoa': 'Chinese', 'món nhật': 'Japanese', 'món hàn': 'Korean',
        'đài loan': 'Taiwanese', 'món thái': 'Thai', 'philippines': 'Filipino',
        'malaysia': 'Malaysian', 'campuchia': 'Cambodian', 'món á': 'Asian', 'singapore': 'Singaporean',
        
        # --- European Cuisine ---
        'ý': 'Italian', 'bánh pizza': 'Italian', 'tây ban nha': 'Spanish',
        'đức': 'German', 'món âu': 'European', 'bắc âu': 'European', 'pháp': 'French',
        
        # --- American Cuisine ---
        'mỹ': 'American', 'châu mỹ': 'American', 'quốc tế': 'International',

        # --- Other ---
        'other': 'Other', 'khác': 'Other',

        # --- Unknown ---
        'unknown': 'Unknown'
    }
    # Hàm lấy tên quốc gia thuộc về
    def get_countries(style_value):
        if pd.isna(style_value): styles = ['unknown']
        else: styles = [s.strip().lower() for s in str(style_value).split(',')]
        # Trả về tập hợp các quốc gia đã map được
        return list({country_mapping[s] for s in styles if s in country_mapping})

    # Tạo danh sách các nhóm quốc gia cho từng dòng
    df['Country_List'] = df[column_name].apply(get_countries)

    # 2. Tạo Multi-Hot Encoding
    mlb = MultiLabelBinarizer()
    encoded_data = mlb.fit_transform(df['Country_List'])
    encoded_df = pd.DataFrame(encoded_data, columns=[f'Style_{c}' for c in mlb.classes_], index=df.index)\

    # 3. Tạo cột Is_Other (Quán không thuộc bất kỳ nhóm nào trong từ điển)
    # Tức là dòng đó không có phong cách nào được map (phát sinh style mới ngoài định nghĩa)
    encoded_df['Style_Other'] = (encoded_df.sum(axis=1) == 0).astype(int)

    # Ghép lại vào dataframe gốc
    df = pd.concat([df, encoded_df], axis=1)
    
    return df.drop(columns=['Country_List'])

# Hàm phân loại món ăn vào nhóm Main, Beverage, Dessert, Snack và gán cờ Vegetarian/Healthy
def extract_cuisine_features(df):
    '''
    Mục tiêu: Phân loại các món ăn theo các nhóm chính
    Kỹ thuật: tạo các đặc trưng nhóm món ăn chính, sử dụng One-hot encoding cho chế độ ăn
    '''
    df = df.copy()
    food_category_mapping = FOOD_CATEGORY_MAPPING

    # Từ điển chế độ ăn (chay, healthy)
    dietary_mapping = {
    'Is_Vegetarian': ['Món chay', 'Đậu hũ'],
    'Is_Healthy': ['Salad', 'Nấm', 'Món chay', 'Trái cây', 'Sinh tố - Nước ép']
    }
    # Khởi tạo các cột
    for category in food_category_mapping.keys():
        df[category] = 0
    for flag in dietary_mapping.keys():
        df[flag] = 0
    
    def count_items(cuisine_str, category):
        # Tách chuỗi món ăn bằng dấu phẩy
        items = [i.strip() for i in str(cuisine_str).split(',')]
        # Đếm xem có bao nhiêu món thuộc category này
        count = sum(1 for item in items if item in food_category_mapping[category])
        return count

    # Điền giá trị
    for category in food_category_mapping.keys():
        df[category] = df['Cuisines'].apply(lambda x: count_items(x, category))
    for flag in dietary_mapping.keys():
        df[flag] = df['Cuisines'].apply(lambda x: 1 if any(item in dietary_mapping[flag] for item in str(x).split(',')) else 0)

    # Tính Variety Score (Số lượng loại món khác nhau mà quán phục vụ)
    # Ví dụ quán có cả Main Dish và Beverage thì score là 2
    df['Cuisines_VarietyScore'] = df[['Cuisines_MainDish', 'Cuisines_Beverage', 'Cuisines_Dessert', 'Cuisines_Snack']].gt(0).sum(axis=1)
    
    return df

# Hàm rút ra các tuple(cuisines, price_bucket) đặc trưng cho từng Type 
def extract_type_price_map(
    df, cuisines_col='Cuisines', type_col='Type', price_col='Avg_Price', tuple_threshold=0.2,
    min_restaurant_count=2, dedup_jaccard=0.7, price_bins=(0, 100_000, 300_000, float('inf')),
    price_labels=('low', 'moderate', 'high'), keep_merge=False
):
    '''
    Rút ra các tuple(cuisines, price_bucket) đặc trưng cho từng Type
    dish_threshold: tỉ lệ xuất hiện tối thiểu để lọc các cuisines không liên quan
    min_restaurant_count: ngưỡng số quán tối thiểu có Type đó, bỏ qua các Type yếu
    price_bins (Mặc định: (0, 100k, 300k, inf)): định nghĩa các khoảng giá.

    '''
    # Tiền xử lý: loại hàng thiếu cột cần thiết
    valid_df = df[
        df[type_col].notna() &
        df[cuisines_col].notna() &
        df[price_col].notna()
    ].copy()

    # Gán price_bucket cho từng hàng
    valid_df['_price_bucket'] = pd.cut(
        valid_df[price_col],
        bins=list(price_bins),
        labels=price_labels,
        right=True,
        include_lowest=True
    ).astype(str)

    type_counts             = Counter()
    tuple_counts_per_type   = defaultdict(Counter)  # {type: {(cuisine, bucket): count}}

    # Bước 1: Đếm tần suất
    for _, row in valid_df.iterrows():
        types   = [t.strip() for t in re.split(r',\s*', str(row[type_col]))   if t.strip()]
        cuisines= [c.lower().strip() for c in re.split(r',\s*', str(row[cuisines_col])) if c.strip()]
        bucket  = row['_price_bucket']

        for typ in types:
            type_counts[typ] += 1
            for cuisine in cuisines:
                key = (cuisine, bucket)
                tuple_counts_per_type[typ][key] += 1

    # Bước 2: Thresholding + lọc Type yếu
    raw_type_map = {}   # {type: set of (cuisine, bucket)}
    raw_freq_map = {}   # {type: {(cuisine, bucket): freq}}

    for typ, total in type_counts.items():
        if total < min_restaurant_count:
            continue
        min_count = total * tuple_threshold
        significant = {
            key: count / total
            for key, count in tuple_counts_per_type[typ].items()
            if count >= min_count
        }
        if significant:
            raw_type_map[typ] = set(significant.keys())
            raw_freq_map[typ] = significant

    # Bước 3: Deduplication bằng Jaccard
    merged = {}
    for t1, t2 in combinations(raw_type_map.keys(), 2):
        s1, s2  = raw_type_map[t1], raw_type_map[t2]
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= dedup_jaccard:
            keep   = t1 if type_counts[t1] >= type_counts[t2] else t2
            remove = t2 if keep == t1 else t1
            merged[remove] = keep

    if not keep_merge:
        type_cuisine_map = {t: d for t, d in raw_type_map.items() if t not in merged}
    else:
        type_cuisine_map = {t: d for t, d in raw_type_map.items()}

    cuisine_freq_map = {t: f for t, f in raw_freq_map.items()}

    return type_cuisine_map, cuisine_freq_map, merged

# Hàm mã hóa cột Type
def encode_type(df, col, prefix=None, drop_original=False):
    from n2_data_cleaning_and_imputation.utils import split_multilabel_cell
    df = df.copy()

    if col not in df.columns:
        return df

    if prefix is None:
        prefix = col

    list_col = f"_{col}_list"
    df[list_col] = df[col].apply(split_multilabel_cell)

    mlb = MultiLabelBinarizer()

    encoded = pd.DataFrame(
        mlb.fit_transform(df[list_col]),
        columns=[f"{prefix}_{label}" for label in mlb.classes_],
        index=df.index
    )

    df = pd.concat([df, encoded], axis=1)
    df = df.drop(columns=[list_col], errors="ignore")

    if drop_original:
        df = df.drop(columns=[col], errors="ignore")

    return df

# Hàm trích xuất các features thời gian
def parse_opening_hours(value):
    if pd.isna(value):
        return pd.Series([float('nan'), float('nan'), float('nan')])

    text = str(value).strip()

    matches = re.findall(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", text)

    if not matches:
        return pd.Series([float('nan'), float('nan'), float('nan')])

    open_hours  = []
    close_hours = []
    total_duration = 0.0

    for open_h, open_m, close_h, close_m in matches:
        o = int(open_h) + int(open_m) / 60
        c = int(close_h) + int(close_m) / 60

        dur = c - o
        if dur < 0:      # qua đêm: 22:00 - 02:00
            dur += 24

        open_hours.append(o)
        close_hours.append(c)
        total_duration += dur

    return pd.Series([min(open_hours), max(close_hours), total_duration])

# Hàm tạo các features mới từ Thời gian của từng ngày trong tuần
def create_opening_hour_features(
    df,
    day_cols=None,
    keep_daily_features=False
):
    """
    Tạo feature từ các cột Monday -> Sunday.
    keep_daily_features=True để tạo thêm các cột:
    - Monday_open
    - Monday_close
    - Monday_duration
    - ...

    Luôn tạo các cột tổng hợp:
    - avg_open_hour
    - avg_close_hour
    - avg_open_duration
    - is_open_weekend
    - is_open_after_22
    """
    df = df.copy()

    if day_cols is None:
        day_cols = DAY_COLS

    existing_days = [col for col in day_cols if col in df.columns]

    for day in existing_days:
        df[[f"{day}_open", f"{day}_close", f"{day}_duration"]] = (
            df[day].apply(parse_opening_hours)
        )

    open_cols = [f"{day}_open" for day in existing_days]
    close_cols = [f"{day}_close" for day in existing_days]
    duration_cols = [f"{day}_duration" for day in existing_days]

    if open_cols:
        df["Avg_open_hour"] = df[open_cols].mean(axis=1)

    if close_cols:
        df["Avg_close_hour"] = df[close_cols].mean(axis=1)

        # Có đóng cửa sau 22h ở bất kỳ ngày nào không?
        df["Is_open_after_22"] = (
            df[close_cols].gt(22).any(axis=1).astype(int)
        )
    else:
        df["Is_open_after_22"] = 0

    if duration_cols:
        df["Avg_open_duration"] = df[duration_cols].mean(axis=1)

    weekend_duration_cols = [
        col for col in ["Saturday_duration", "Sunday_duration"]
        if col in df.columns
    ]

    if weekend_duration_cols:
        df["Is_open_weekend"] = (
            df[weekend_duration_cols].notna().any(axis=1).astype(int)
        )
    else:
        df["Is_open_weekend"] = 0

    if not keep_daily_features:
        daily_created_cols = []
        for day in existing_days:
            daily_created_cols.extend([
                f"{day}_open",
                f"{day}_close",
                f"{day}_duration"
            ])

        df = df.drop(columns=daily_created_cols, errors="ignore")

    return df
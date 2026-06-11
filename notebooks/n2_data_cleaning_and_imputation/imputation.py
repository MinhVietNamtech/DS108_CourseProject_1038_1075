import numpy as np
import pandas as pd
import re
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

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

# Hàm bổ sung Cuisines từ tên quán (nếu có)
def impute_cuisines_from_name(df, name_col='Restaurant Name', cuisine_col='Cuisines'):
    """
    Điền khuyết Cuisines bằng cách quét từ khóa món ăn trong tên quán.
    Sử dụng FOOD_CATEGORY_MAPPING để tìm các món ăn đặc trưng.
    """
    df_clean = df.copy()

    # Lấy tất cả các món ăn từ mọi category trong FOOD_CATEGORY_MAPPING
    all_keywords = []
    for cat_list in FOOD_CATEGORY_MAPPING.values():
        all_keywords.extend(cat_list)
    # Loại bỏ trùng lặp và sắp xếp theo độ dài giảm dần (ưu tiên cụm từ dài hơn)
    all_keywords = sorted(list(set(all_keywords)), key=len, reverse=True)

    def find_cuisines(row):
        # Chỉ điền nếu Cuisines bị thiếu
        if pd.notna(row[cuisine_col]) and str(row[cuisine_col]).strip() != '':
            return row[cuisine_col]
        
        name = row[name_col]
        # Kiểm tra nếu tên quán bị thiếu hoặc chỉ toàn khoảng trắng
        if pd.isna(name) or str(name).strip() == '':
            return row[cuisine_col]
        
        name = str(name).lower()
        found = []
        for kw in all_keywords:
            if kw.lower() in name:
                found.append(kw)

        if found:
            return ", ".join(found)
        return row[cuisine_col]

    df_clean[cuisine_col] = df_clean.apply(find_cuisines, axis=1)
    return df_clean

# Hàm điền giá trị thiếu cột Style
def impute_missing_styles_overlap(df, style_dish_map, threshold=0.1, top_k=3):
    """
    Điền Style sử dụng Overlap Coefficient (Szymkiewicz-Simpson).
    Công thức: score = |A ∩ B| / min(|A|, |B|)
    """
    df_clean = df.copy()
    missing_mask = df_clean['Style'].isna() & df_clean['Cuisines'].notna()

    imputation_log = []

    for idx, row in df_clean[missing_mask].iterrows():
        row_dishes = {d.strip().lower() for d in re.split(r'[,]+', str(row['Cuisines'])) if d.strip()}
        if not row_dishes:
            df_clean.at[idx, 'Style'] = 'Other'
            continue

        scores = []
        for style, style_dishes in style_dish_map.items():
            intersection = row_dishes & style_dishes
            if not intersection:
                continue

            # Overlap Coefficient: chia cho kích thước tập nhỏ nhất
            score = len(intersection) / min(len(row_dishes), len(style_dishes))
            if score >= threshold:
                scores.append((style, round(score, 3)))
        if scores:
            scores.sort(key=lambda x: -x[1])
            top_styles = [s for s, _ in scores[:top_k]]

            df_clean.at[idx, 'Style'] = ','.join(top_styles)
            imputation_log.append({
                'idx': idx+1,
                'assigned': ','.join(top_styles),
                'score': scores[:top_k],
                'status': 'filled'
            })
        else:
            df_clean.at[idx, 'Style'] = 'Other'
            imputation_log.append({'idx': idx+1, 'assigned': 'Other', 'score': [], 'status': 'other'})

    return df_clean, imputation_log

# Hàm điền giá trị thiếu cột Cuisines
def impute_missing_cuisines(df, dish_freq_map, cuisines_col = 'Cuisines', style_col = 'Style', freq_threshold=0.1, top_n=4):
    df_clean     = df.copy()
    missing_mask = df_clean[cuisines_col].isna() & df_clean[style_col].notna()
    log          = []
 
    for idx, row in df_clean[missing_mask].iterrows():
        styles          = [s.strip() for s in str(row[style_col]).split(',') if s.strip() and str(row[style_col]).lower() != 'nan']
        matched_styles  = [s for s in styles if s in dish_freq_map]
        unknown_styles  = [s for s in styles if s not in dish_freq_map]
 
        if not matched_styles:
            log.append({'idx': idx, 'restaurant': row['Restaurant Name'],
                        'style': str(row[style_col]).strip(), 'assigned': None,
                        'unknown_styles': unknown_styles, 'status': 'skipped'})
            continue
 
        # Cộng dồn tần suất từ tất cả Style khớp
        combined = Counter()
        for style in matched_styles:
            for dish, freq in dish_freq_map[style].items():
                combined[dish] += freq
 
        # Lọc theo tần suất trung bình, lấy top_n
        # Case-sensitive: giữ nguyên cách viết gốc (title case) từ dữ liệu
        top_dishes = [
            dish for dish, score in combined.most_common()
            if (score / len(matched_styles)) >= freq_threshold
        ][:top_n]
 
        if top_dishes:
            # Khôi phục đúng case như trong dữ liệu gốc (impute_style dùng lower())
            cuisine_str = ",".join(d.title() for d in top_dishes)
            df_clean.at[idx, cuisines_col] = cuisine_str
            log.append({'idx': idx+1, 'restaurant': row['Restaurant Name'],
                        'style': str(row[style_col]).strip(), 'assigned': cuisine_str,
                        'unknown_styles': unknown_styles, 'status': 'filled'})
        else:
            log.append({'idx': idx+1, 'restaurant': row['Restaurant Name'],
                        'style': str(row[style_col]).strip(), 'assigned': None,
                        'unknown_styles': unknown_styles, 'status': 'skipped'})
 
    return df_clean, log

# Hàm điền giá trị thiếu cột mục tiêu bằng Mode theo khu vực 
# (nhóm theo Tỉnh/Thành -> Quận/Huyện)
def impute_by_location(df, city_col='City', district_col='District', target_col='Cuisines'):
    """
    Điền khuyết Cuisines bằng cách lấy giá trị phổ biến nhất (mode)
    của các quán trong cùng Tỉnh và Quận/Huyện.
    """
    df_clean = df.copy()

    # 1. Tính mode toàn cục để làm fallback
    global_mode = df_clean[target_col].mode()
    global_val = global_mode[0] if not global_mode.empty else 'Unknown'

    # 2. Tính mode theo từng nhóm City và District
    group_mode = df_clean.groupby([city_col, district_col])[target_col].transform(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )

    # 3. Điền giá trị: Ưu tiên mode khu vực -> sau đó đến mode toàn cục
    df_clean[target_col] = df_clean[target_col].fillna(group_mode).fillna(global_val)

    return df_clean

# Hàm điền giá trị thiếu cột Type
# Chia Price_bucket thành các khoảng: 
# <100.000đ: bình dân, <300.000đ: trung cấp, >=300.000đ: cao cấp 
def impute_missing_type(df, type_price_map, threshold=0.1, top_k=2, price_col='Avg_Price',
                        price_bins=(0, 100_000, 300_000, float('inf')),
                        price_labels=('low', 'moderate', 'high')):

    df_clean = df.copy()
    missing_mask = (
        df_clean['Type'].isna() &
        df_clean['Cuisines'].notna() &
        df_clean[price_col].notna()
    )

    imputation_log = []

    for idx, row in df_clean[missing_mask].iterrows():
        # Tạo set tuple (cuisine, price_bucket) từ row 
        cuisines = {c.strip().lower() for c in re.split(r'[,]+', str(row['Cuisines'])) if c.strip()}
        if not cuisines:
            df_clean.at[idx, 'Type'] = 'Other'
            continue

        bucket = pd.cut(
            [row[price_col]],
            bins=list(price_bins),
            labels=price_labels,
            right=True,
            include_lowest=True
        )[0] # lấy label duy nhất
        bucket = str(bucket)

        row_tuples = {(cuisine, bucket) for cuisine in cuisines}  # so sánh với type_price_map

        # Tính Overlap Coefficient
        scores = []
        for typ, type_tuples in type_price_map.items():
            intersection = row_tuples & type_tuples
            if not intersection:
                continue

            score = len(intersection) / min(len(row_tuples), len(type_tuples))
            if score >= threshold:
                scores.append((typ, round(score, 3)))

        # Gán kết quả
        if scores:
            scores.sort(key=lambda x: -x[1])
            top_types = [t for t, _ in scores[:top_k]]

            df_clean.at[idx, 'Type'] = ','.join(top_types)
            imputation_log.append({
                'idx'     : idx + 1,
                'assigned': ','.join(top_types),
                'score'   : scores[:top_k],
                'status'  : 'filled'
            })
        else:
            df_clean.at[idx, 'Type'] = 'Other'
            imputation_log.append({
                'idx'     : idx + 1,
                'assigned': 'Other',
                'score'   : [],
                'status'  : 'other'
            })

    return df_clean, imputation_log

# Class điền giá thiếu
class PriceImputer:
    def __init__(self, target_cols=('Min_Price', 'Max_Price'), min_group_size=10):
        self.target_cols = list(target_cols)
        self.min_group_size = min_group_size
        self.global_map = None
        self.local_maps = {} # Lưu trữ bản đồ theo (City, District)

    def _extract_cuisines_price(self, df_scope):
        """Hàm helper để tính giá trung bình theo cuisine (dùng chung cho fit)."""
        df_valid = df_scope[df_scope[self.target_cols].notna().any(axis=1) & df_scope['Cuisines'].notna()].copy()
        if df_valid.empty:
            return pd.DataFrame(columns=self.target_cols)

        df_exploded = df_valid.assign(Cuisine=df_valid['Cuisines'].str.split(',')).explode('Cuisine')
        df_exploded['Cuisine'] = df_exploded['Cuisine'].str.strip()

        return df_exploded.groupby('Cuisine')[self.target_cols].mean().round(0)

    def fit(self, df, city_col='City', district_col='District'):
        """Học các bản đồ giá từ tập Train."""
        # 1. Fit Global Map
        self.global_map = self._extract_cuisines_price(df)

        # 2. Fit Local Maps
        self.local_maps = {}
        for (city, district), group in df.groupby([city_col, district_col], dropna=False):
            if len(group) >= self.min_group_size:
                self.local_maps[(city, district)] = self._extract_cuisines_price(group)
        
        return self

    def transform(self, df, city_col='City', district_col='District'):
        """Áp dụng bản đồ đã học để điền giá trị thiếu (cho cả Train và Test)."""
        result = df.copy()

        def impute_row(row):
            # 1. Nếu đã có đủ giá, trả về luôn
            if all(pd.notna(row[col]) for col in self.target_cols):
                return row[self.target_cols]
            
            if pd.isna(row['Cuisines']):
                return row[self.target_cols]

            cuisines = [c.strip() for c in row['Cuisines'].split(',')]
            
            # 2. Lookup theo thứ tự ưu tiên: Local -> Global
            key = (row[city_col], row[district_col])
            maps_to_try = [self.local_maps.get(key), self.global_map]
            
            for price_map in maps_to_try:
                if price_map is None or price_map.empty: continue
                
                matched = [c for c in cuisines if c in price_map.index]
                if not matched: continue
                
                subset = price_map.loc[matched]
                
                # Lấy Min nhỏ nhất và Max lớn nhất từ các món tìm được
                min_val = subset['Min_Price'].min()
                max_val = subset['Max_Price'].max()
                
                # 3. Cơ chế suy luận (Fallback logic)
                # Nếu chỉ tìm được 1 trong 2, thì gán giá trị kia bằng giá trị đã tìm được
                if pd.notna(min_val) and pd.isna(max_val):
                    max_val = min_val
                elif pd.isna(min_val) and pd.notna(max_val):
                    min_val = max_val
                
                # Nếu tìm được cả hai, trả về kết quả
                if pd.notna(min_val) or pd.notna(max_val):
                    return pd.Series([float(min_val), float(max_val)], index=self.target_cols)

            # Nếu không tìm thấy bằng Lookup, giữ nguyên giá trị cũ (NaN hoặc có sẵn)
            return row[self.target_cols]

        # Áp dụng logic điền vào
        result[self.target_cols] = df.apply(impute_row, axis=1)

        # 4. Đảm bảo Max >= Min cuối cùng
        swap_mask = (result['Max_Price'] < result['Min_Price']) & result[self.target_cols].notna().all(axis=1)
        result.loc[swap_mask, ['Min_Price', 'Max_Price']] = result.loc[swap_mask, ['Max_Price', 'Min_Price']].values
        
        return result

# Hàm điền giá trị thiếu các cột về Thời gian (đã biến đổi)
def impute_missing_time(df, city_col='City', district_col='District',
                         type_col='Type', target_col='Avg_open_hour'):
    df_cleaned = df.copy()
    group_cols = [
        [city_col, district_col, type_col],
        [city_col, district_col],
        [city_col] # global
    ]
    for grp in group_cols:
        group_mode = df_cleaned.groupby(grp)[target_col].transform(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        )
        df_cleaned[target_col] = df_cleaned[target_col].fillna(group_mode)
    # Fallback cuối cùng: global mode
    global_mode = df_cleaned[target_col].mode()
    if not global_mode.empty:
        df_cleaned[target_col] = df_cleaned[target_col].fillna(global_mode.iloc[0])

    return df_cleaned


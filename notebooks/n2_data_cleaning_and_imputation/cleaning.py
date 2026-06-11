import numpy as np
import pandas as pd
import re
from rapidfuzz import fuzz
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# List các type không liên quan đến đề tài
EXCLUDED_TYPES = {
    "du lịch sinh thái", "karaoke", "thăm quan & chụp ảnh", "khách sạn",
    "khu nghi dưỡng", "rạp chiếu phim", "shop/cửa hàng", "spa/massage",
    "tàu du lịch", "tiệc tận nơi", "chùa & nhà thờ", "trung tâm thương mại",
    "khu chơi game", "chợ", "thể dục thể thao", "homestay", "mua sắm online",
    "nhà sách & thư viện", "chụp hình cưới", "công viên vui chơi", "khu nghỉ dưỡng", "trường dạy nghề",
    "billiards", "trang điểm", "thể dục thẩm mỹ", "tiệc cưới/hội nghị", "bar/pub", "beer club"
}

# Hàm lọc các types không liên quan đến đề tài
def remove_excluded_types(df, col="Type"):
    def has_excluded(value):
        if pd.isna(value):
            return False
        parts = [p.strip().lower() for p in str(value).split(',')]
        return any(p in EXCLUDED_TYPES for p in parts)
    
    mask = df[col].apply(has_excluded)
    return df[~mask].reset_index(drop=True)

# Hàm chuẩn hóa giá trị đa nhãn
def normalize_list_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return text

    # Chuyển thành lowercase và tách theo dấu phẩy
    parts = str(text).lower().split(',')
    # Xóa khoảng trắng và lọc bỏ chuỗi rỗng
    cleaned = [p.strip() for p in parts if p.strip()]

    return ', '.join(cleaned) if cleaned else np.nan

# Hàm chuẩn hóa văn bản cho các cột có giá trị đa nhãn (Style, Cuisines,Types)
def normalize_categorical_columns(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_list_text)
    return df

# Hàm chuẩn hóa tên quận/huyện
def normalize_districts(df, district_col='District'):
    df = df.copy()
    df[district_col] = df[district_col].fillna('').str.strip().str.lower()

    def normalize_name(name: str) -> str:
        """
        Bỏ tiền tố hành chính, chỉ giữ tên riêng.
        """
        if not name or not isinstance(name, str):
            return ''
        text = name.strip().lower()
        # Giữ nguyên quận số (quận 1 -> quận 12)
        if re.fullmatch(r'qu[aậ]n\s*\d+', text):
            return re.sub(r'qu[aậ]n\s*', 'quận ', text)

        # Bỏ tiền tố hành chính
        prefixes = r'^(quận|huyện|thị\s*xã|thành\s*phố|tp\.?)\s*'
        text = re.sub(prefixes, '', text).strip()
        return text
    
    df[district_col] = df[district_col].apply(normalize_name)
    return df

# Hàm phát hiện quán trùng sử dụng fuzzy matching với ngưỡng 90% cho
# cột Restaurant Name và Address
def find_fuzzy_duplicates(group, name_col, addr_col,
                           name_threshold=90, addr_threshold=90):
    """
    Tìm các cặp trùng gần đúng trong nhóm theo 2 tiêu chí:
      - Tên giống >= name_threshold%
      - Địa chỉ giống >= addr_threshold%
    Chỉ đánh dấu trùng khi thoả cả 2/2 tiêu chí.
    """
    drop_idx = set()
    valid = group.dropna(subset=[name_col])

    for i, j in combinations(valid.index, 2):
        if i in drop_idx or j in drop_idx:
            continue

        row_i, row_j = valid.loc[i], valid.loc[j]
        matches = 0

        # Tiêu chí 1: tên giống
        name_score = fuzz.token_sort_ratio(
            str(row_i[name_col]).lower().strip(),
            str(row_j[name_col]).lower().strip()
        )
        if name_score >= name_threshold:
            matches += 1

        # Tiêu chí 2: địa chỉ giống
        if pd.notna(row_i[addr_col]) and pd.notna(row_j[addr_col]):
            addr_score = fuzz.token_sort_ratio(
                str(row_i[addr_col]).lower().strip(),
                str(row_j[addr_col]).lower().strip()
            )
            if addr_score >= addr_threshold:
                matches += 1

        if matches >= 2:
            drop_idx.add(j)

    return drop_idx

# Hàm lọc trùng quán theo 3 lớp: 
# 1. Trùng cả Restaurant Name + Latitude + Longitude, 2. Trùng Latitude + Longitude, 3. Trùng Address
# Có sử dụng fuzzy mathching để lọc tên và địa chỉ quán với ngưỡng 90%
def drop_duplicates_location(df, name_col='Restaurant Name',
                              lat_col='Latitude', lon_col='Longitude',
                              addr_col='Address',
                              city_col='City', district_col='District',
                              keep='first', fuzzy=False,
                              name_threshold=90, addr_threshold=90):
    before = len(df)
    drop_indices = set()

    for _, group in df.groupby([city_col, district_col], dropna=False):
        mask_has_coords = group[lat_col].notna() & group[lon_col].notna()
        mask_has_name   = group[name_col].notna() & group[name_col].str.strip().ne('')
        mask_has_addr   = group[addr_col].notna() & group[addr_col].str.strip().ne('')

        # Lớp 1: trùng tên + toạ độ
        mask_l1 = mask_has_name & mask_has_coords
        dup_l1  = group[mask_l1].duplicated(subset=[name_col, lat_col, lon_col], keep=keep)
        drop_indices.update(dup_l1[dup_l1].index)
        group = group[~group.index.isin(drop_indices)]

        # Lớp 2: trùng tên + địa chỉ
        mask_has_name = group[name_col].notna() & group[name_col].str.strip().ne('')
        mask_has_addr = group[addr_col].notna() & group[addr_col].str.strip().ne('')
        mask_l2 = mask_has_name & mask_has_addr
        dup_l2  = group[mask_l2].duplicated(subset=[name_col, addr_col], keep=keep)
        drop_indices.update(dup_l2[dup_l2].index)
        group = group[~group.index.isin(drop_indices)]

        # Lớp 3: trùng toạ độ
        mask_has_coords = group[lat_col].notna() & group[lon_col].notna()
        dup_l3 = group[mask_has_coords].duplicated(subset=[lat_col, lon_col], keep=keep)
        drop_indices.update(dup_l3[dup_l3].index)
        group = group[~group.index.isin(drop_indices)]

        # Lớp 4: trùng tên
        mask_has_name = group[name_col].notna() & group[name_col].str.strip().ne('')
        dup_l4 = group[mask_has_name].duplicated(subset=[name_col], keep=keep)
        drop_indices.update(dup_l4[dup_l4].index)
        group = group[~group.index.isin(drop_indices)]

        # Lớp 5: fuzzy tên + địa chỉ
        if fuzzy:
            mask_has_name = group[name_col].notna() & group[name_col].str.strip().ne('')
            fuzzy_drop = find_fuzzy_duplicates(
                group[mask_has_name], name_col, addr_col,
                name_threshold, addr_threshold
            )
            drop_indices.update(fuzzy_drop)

    df = df[~df.index.isin(drop_indices)]
    print(f"Trước: {before} hàng  |  Sau: {len(df)} hàng  |  Đã xoá: {before - len(df)} hàng")
    return df.reset_index(drop=True)

# Hàm hiện các dòng trùng của từng lớp (có thể trùng lặp giữa các lớp)
def show_duplicates_location(df, name_col='Restaurant Name', lat_col='Latitude',
                            lon_col='Longitude', city_col='City', district_col='District'):
    display_cols = [name_col, lat_col, lon_col]
    results = {'Lớp 1 (name + toạ độ)': [], 'Lớp 2 (toạ độ)': [], 'Lớp 3 (name)': []}

    for (city, district), group in df.groupby([city_col, district_col]):
        mask_has_name   = group[name_col].notna() & group[name_col].str.strip().ne('')
        mask_has_coords = group[lat_col].notna() & group[lon_col].notna()

        # Lớp 1: trùng tên + toạ độ
        dup_l1 = group[mask_has_name & mask_has_coords].duplicated(
            subset=[name_col, lat_col, lon_col], keep=False)
        results['Lớp 1 (name + toạ độ)'].append(
            group[mask_has_name & mask_has_coords & dup_l1]
            .sort_values([name_col, lat_col, lon_col])
        )

        # Lớp 2: trùng toạ độ
        dup_l2 = group[mask_has_coords].duplicated(subset=[lat_col, lon_col], keep=False)
        results['Lớp 2 (toạ độ)'].append(
            group[mask_has_coords & dup_l2]
            .sort_values([lat_col, lon_col])
        )

        # Lớp 3: trùng tên
        dup_l3 = group[mask_has_name].duplicated(subset=[name_col], keep=False)
        results['Lớp 3 (name)'].append(
            group[mask_has_name & dup_l3]
            .sort_values(name_col)
        )

    # Gộp tất cả nhóm lại
    results = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=display_cols)
               for k, v in results.items()}

    print("Số hàng trùng theo từng lớp:")

    output_strings = []
    for level, dup_df in results.items():
        print(f"\n{'='*60}")
        print(f"{level}: {len(dup_df)} hàng")
        print('='*60)
        if dup_df.empty:
            print("  Không có trùng lặp")
        else:
            output_strings.append(dup_df[display_cols].to_string(index=True))

    return results, output_strings

class PriceOutlierHandler:
    def __init__(self, multiplier=1.5, min_group_size=10):
        self.multiplier = multiplier
        self.min_group_size = min_group_size
        self.thresholds = {}  # Lưu trữ các ngưỡng đã fit

    def _cap_values(self, series, lo, hi):
        return series.clip(lower=lo, upper=hi)

    def fit(self, df, min_col='Min_Price', max_col='Max_Price', city_col='City', district_col='District'):
        self.thresholds = {}
        processed_idx = set()
        
        # Helper tìm ngưỡng
        def get_thresholds(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = max(q1 - self.multiplier * iqr, 0)
            upper = q3 + self.multiplier * iqr
            return lower, upper

        # 1. Fit Cấp 1: City + District
        for (city, district), grp in df.groupby([city_col, district_col]):
            idx = grp[grp[min_col].notna() | grp[max_col].notna()].index
            if len(idx) >= self.min_group_size:
                lo_min, hi_min = get_thresholds(grp[min_col].dropna())
                lo_max, hi_max = get_thresholds(grp[max_col].dropna())
                self.thresholds[f"{city}_{district}"] = (lo_min, hi_min, lo_max, hi_max)
                processed_idx.update(idx)

        # 2. Fit Cấp 2: City (fallback)
        remaining = df[~df.index.isin(processed_idx)]
        for city, grp in remaining.groupby(city_col):
            idx = grp[grp[min_col].notna() | grp[max_col].notna()].index
            if len(idx) >= 3:
                lo_min, hi_min = get_thresholds(grp[min_col].dropna())
                lo_max, hi_max = get_thresholds(grp[max_col].dropna())
                self.thresholds[f"{city}_fallback"] = (lo_min, hi_min, lo_max, hi_max)
                processed_idx.update(idx)
        
        return self

    def transform(self, df, min_col='Min_Price', max_col='Max_Price', city_col='City', district_col='District'):
        """Áp dụng các ngưỡng đã học lên tập dữ liệu."""
        df = df.copy()
        
        def apply_row(row):
            key = f"{row[city_col]}_{row[district_col]}"
            if key in self.thresholds:
                lo_min, hi_min, lo_max, hi_max = self.thresholds[key]
            else:
                key_fallback = f"{row[city_col]}_fallback"
                if key_fallback in self.thresholds:
                    lo_min, hi_min, lo_max, hi_max = self.thresholds[key_fallback]
                else:
                    return row # Không có ngưỡng thì giữ nguyên
            
            row[min_col] = np.clip(row[min_col], lo_min, hi_min)
            row[max_col] = np.clip(row[max_col], lo_max, hi_max)
            return row

        df = df.apply(apply_row, axis=1)

        # Đảm bảo Max >= Min
        swap_mask = df[max_col].notna() & df[min_col].notna() & (df[max_col] < df[min_col])
        df.loc[swap_mask, [min_col, max_col]] = df.loc[swap_mask, [max_col, min_col]].values
        
        return df
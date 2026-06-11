from pathlib import Path
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Bảng màu pastel dùng xuyên suốt notebook
PASTEL_COLORS = {
        "foody": "#A8DADC",      # xanh pastel
        "shopee": "#F4A261",     # cam pastel
        "comment": "#50A2A4",
        "restaurant": "#F6BD60",
        "hist": "#CDB4DB",       # tím pastel
        "accent": "#84A59D",     # xanh xám pastel
        "target": "#E55A5A",
        "primary": "#B8E0D2",
        "secondary": "#F7C59F",
        "accent": "#E4C1F9",
        "pink": "#FDE2E4",
        "blue": "#A9DEF9",
        "green": "#CDEAC0",
        "gray": "#D6D6D6",
        "yellow": "#FFF1A8"
    }

# Hàm thiết lập cấu hình
def set_configs():
    # Cấu hình hiển thị biểu đồ cho đẹp hơn

    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["font.family"] = "DejaVu Sans"

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_colwidth", 150)

    plt.rcParams["figure.figsize"] = (10, 5)
    return None

# Hàm vẽ biểu đồ cột ngang đẹp hơn
def plot_barh_beautiful(
    data,
    category_col,
    value_col,
    title,
    xlabel,
    ylabel,
    color="#B8E0D2",
    top_n=None,
    value_format="{:,.0f}",
    save_img=False,
    prefix = "new"
):
    """
    Vẽ biểu đồ cột ngang gọn, dễ đọc, dùng màu pastel.
    """

    plot_data = data.copy()

    if top_n is not None:
        plot_data = plot_data.sort_values(value_col, ascending=False).head(top_n)

    plot_data = plot_data.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_data))))

    bars = ax.barh(
        plot_data[category_col].astype(str),
        plot_data[value_col],
        color=color,
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            value_format.format(width),
            va="center",
            ha="left",
            fontsize=9
        )

    plt.tight_layout()
    if save_img:
        plt.savefig(f"../../images/{prefix}_barh_{category_col}_{value_col}.png", dpi=300, bbox_inches="tight")
    plt.show()

# Hàm vẽ histogram cho biến liên tục và giới hạn khoảng hiển thị theo percentile.
def plot_hist_percentile(
    ax,
    data,
    col,
    title=None,
    color="#A9DEF9",
    bins=30,
    lower_q=0.01,
    upper_q=0.99,
):

    series = data[col].dropna()

    if series.empty:
        ax.text(0.5, 0.5, f"{col} rỗng", ha="center")
        return

    lower_limit = series.quantile(lower_q)
    upper_limit = series.quantile(upper_q)

    if lower_limit != upper_limit:
        series = series[
            (series >= lower_limit)
            & (series <= upper_limit)
        ]

    ax.hist(
        series,
        bins=bins,
        color=color,
        edgecolor="black",
        alpha=0.85
    )

    # ===== title đẹp hơn =====
    pretty_title = (
        title
        if title is not None
        else col.replace("_", " ")
    )

    ax.set_title(
        pretty_title,
        fontsize=12,
        fontweight="bold"
    )

    # ===== bỏ xlabel/ylabel =====
    ax.set_xlabel("")
    ax.set_ylabel("")

    # ===== format số =====
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, p: f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"
        )
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hàm vẽ giá trị trung bình của một feature theo từng mức độ yêu thích bằng line plot.
def plot_line_by_target(
    data,
    feature_col,
    target_col,
    title=None,
    ylabel=None,
    color="#84A59D",
    value_fmt="{:.2f}",
    ax=None  # Thêm tham số ax
):
    mean_by_target = (
        data
        .groupby(target_col)[feature_col]
        .mean()
        .reset_index()
        .sort_values(target_col)
    )

    x_labels = mean_by_target[target_col].astype(str)
    y_values = mean_by_target[feature_col]

    # Nếu không truyền ax vào, tự tạo mới (giúp hàm vẫn chạy độc lập được)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(
        x_labels,
        y_values,
        marker="o",
        markersize=8,
        linewidth=2.2,
        color=color
    )

    for x, y in zip(x_labels, y_values):
        ax.text(
            x,
            y,
            value_fmt.format(y),
            ha="center",
            va="bottom" if y >= y_values.mean() else "top",
            fontsize=9
        )

    if title is None:
        title = f"Giá trị trung bình của {feature_col} theo {target_col}"

    if ylabel is None:
        ylabel = f"Mean {feature_col}"

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(target_col)
    ax.set_ylabel(ylabel)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min = y_values.min()
    y_max = y_values.max()
    padding = max((y_max - y_min) * 0.2, 0.05)
    ax.set_ylim(y_min - padding, y_max + padding)

# Hàm vẽ phân phối lớp
def plot_target(target_col, target_distribution):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        target_distribution[target_col].astype(str),
        target_distribution["count"],
        color=PASTEL_COLORS["target"],
        edgecolor="black",
        linewidth=0.6
    )

    plt.title("Phân bố biến mục tiêu preference_level", fontweight="bold", pad=12)
    plt.xlabel("Preference level")
    plt.ylabel("Số lượng mẫu")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

# Hàm vẽ tổng số dòng theo từng nguồn: Shopee Food, Foody và thể loại (Data, Comment)
def plot_records_per_source(file_summary_df):
    rows_by_source_type = (
        file_summary_df
        .groupby(["source", "data_type"])["num_rows"]
        .sum()
        .reset_index()
    )

    pivot_rows = rows_by_source_type.pivot(
        index="source",
        columns="data_type",
        values="num_rows"
    )

    ax = pivot_rows.plot(
        kind="bar",
        figsize=(8, 5),
        color=[PASTEL_COLORS["comment"], PASTEL_COLORS["restaurant"]],
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_title("Tổng số dòng theo nguồn và loại dữ liệu", fontweight="bold", pad=12)
    ax.set_xlabel("Nguồn dữ liệu")
    ax.set_ylabel("Tổng số dòng")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Gắn nhãn số lên đầu cột
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.show()

# Hàm vẽ tỷ lệ feature nhị phân = 1 theo từng lớp target.
def plot_binary_ratio_by_target(
    data, 
    feature_col, 
    target_col, 
    color="#E4C1F9",
    ax=None  # Thêm tham số ax
):
    """
    Vẽ tỷ lệ feature nhị phân = 1 theo từng lớp target.
    """
    ratio_by_target = (
        data
        .groupby(target_col)[feature_col]
        .mean()
        .reset_index()
        .sort_values(target_col)
    )

    ratio_by_target[feature_col] = ratio_by_target[feature_col] * 100

    x_labels = ratio_by_target[target_col].astype(str)
    y_values = ratio_by_target[feature_col]

    # Kiểm tra nếu ax không được truyền vào thì tự tạo
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(
        x_labels,
        y_values,
        marker="o",
        markersize=8,
        linewidth=2.2,
        color=color
    )

    for x, y in zip(x_labels, y_values):
        ax.text(x, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Tỷ lệ {feature_col}=1\ntheo {target_col}", fontweight="bold", pad=12)
    ax.set_xlabel(target_col)
    ax.set_ylabel("Tỷ lệ xuất hiện (%)")

    ax.set_ylim(0, min(100, max(y_values.max() * 1.25, 5)))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hàm lấy danh sách tất cả file dữ liệu trong một thư mục.
def get_data_files(folder):
    """    
    Hỗ trợ file csv, xlsx và xls.
    """
    
    files = (
        list(folder.glob("*.csv")) +
        list(folder.glob("*.xlsx")) +
        list(folder.glob("*.xls"))
    )
    
    return sorted(files)

# Hàm đọc file dữ liệu theo định dạng file.
def read_data_file(file_path):
    """    
    Nếu file csv/excel bị rỗng hoặc không đọc được dữ liệu,
    hàm sẽ trả về DataFrame rỗng để notebook không bị dừng.
    """
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == ".csv":
            return pd.read_csv(file_path)
        
        elif file_extension in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        
        else:
            raise ValueError(f"Định dạng file chưa được hỗ trợ: {file_extension}")
    
    except pd.errors.EmptyDataError:
        # Trường hợp file csv rỗng hoàn toàn, không có cột để đọc
        return pd.DataFrame()

# Hàm lấy mã khu vực từ tên file.
def extract_area_code(file_path):
    """    
    Ví dụ:
    C108.csv -> 108
    D108.csv -> 108
    """
    file_name = file_path.stem
    
    if file_name[0].upper() in ["C", "D"]:
        return file_name[1:]
    
    return file_name

# Hàm vẽ phân bố số từ trong comment 
def comment_hist(raw_comments):
    plt.figure(figsize=(10, 5))

    plt.hist(
        raw_comments["comment_word_count"].dropna(),
        bins=40,
        color=PASTEL_COLORS["hist"],
        edgecolor="black",
        alpha=0.85
    )

    plt.title("Phân bố số từ trong comment", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Số từ")
    plt.ylabel("Số lượng comment")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    return None

def count_outliers_iqr(series):
    """
    Đếm outlier theo IQR.
    Nếu IQR = 0, không tự động xem các giá trị khác mode là outlier,
    vì nhiều feature thưa/zero-inflated có phân phối tập trung ở 0.
    """
    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return {
            "outlier_count": np.nan,
            "outlier_ratio": np.nan,
            "lower_bound": q1,
            "upper_bound": q3,
            "iqr": iqr,
            "note": "IQR = 0; không diễn giải outlier theo IQR cho feature này"
        }

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()

    return {
        "outlier_count": int(outlier_count),
        "outlier_ratio": round(outlier_count / len(series) * 100, 2),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "iqr": iqr,
        "note": "OK"
    }

def status_summary(raw_restaurants, raw_comments):
    # 1. Comment không liên kết được với quán
    restaurant_keys = set(
        raw_restaurants[["source", "area_code", "RestaurantID"]]
        .dropna()
        .astype(str)
        .apply(tuple, axis=1)
    )

    comment_keys = (
        raw_comments[["source", "area_code", "RestaurantID"]]
        .dropna()
        .astype(str)
        .apply(tuple, axis=1)
    )

    comments_without_restaurant = raw_comments[
        ~comment_keys.isin(restaurant_keys)
    ]

    # 2. Quán không có comment
    comment_restaurant_keys = set(
        raw_comments[["source", "area_code", "RestaurantID"]]
        .dropna()
        .astype(str)
        .apply(tuple, axis=1)
    )

    restaurant_keys_series = (
        raw_restaurants[["source", "area_code", "RestaurantID"]]
        .dropna()
        .astype(str)
        .apply(tuple, axis=1)
    )

    restaurants_without_comments = raw_restaurants[
        ~restaurant_keys_series.isin(comment_restaurant_keys)
    ]

    # 3. Comment trùng có thể loại bỏ
    comment_duplicate_cols = [
        "source",
        "area_code",
        "RestaurantID",
        "UserID",
        "Comment"
    ]

    comment_duplicate_cols = [
        col for col in comment_duplicate_cols
        if col in raw_comments.columns
    ]

    duplicate_comments_to_remove = raw_comments.duplicated(
        subset=comment_duplicate_cols,
        keep="first"
    ).sum()

    # 4. Quán trùng theo tên và địa chỉ
    restaurant_duplicate_cols = [
        "Restaurant Name",
        "Latitude",
        "Longitude"
    ]

    restaurant_duplicate_cols = [
        col for col in restaurant_duplicate_cols
        if col in raw_restaurants.columns
    ]

    duplicate_restaurants = raw_restaurants[
        raw_restaurants.duplicated(
            subset=restaurant_duplicate_cols,
            keep=False
        )
    ]

    # 5. Comment quá ngắn
    if "comment_word_count" not in raw_comments.columns:
        raw_comments["comment_word_count"] = (
            raw_comments["Comment"]
            .astype(str)
            .str.split()
            .str.len()
        )

    short_comments = raw_comments[
        raw_comments["comment_word_count"] <= 2
    ]

    print("Số comment không liên kết được với quán:", comments_without_restaurant.shape[0])
    print("Số quán không có comment:", restaurants_without_comments.shape[0])
    print("Số comment trùng có thể loại bỏ:", duplicate_comments_to_remove)
    print("Số quán trùng theo tên và địa chỉ:", duplicate_restaurants.shape[0])
    print("Số comment quá ngắn:", short_comments.shape[0])
    return None
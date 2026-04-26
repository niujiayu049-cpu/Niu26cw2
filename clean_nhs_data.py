from pathlib import Path
from collections import defaultdict
import zipfile
import re
import pandas as pd


zip_path = Path("OneDrive_2026-04-26.zip")
extract_dir = Path("nhs_extracted")
data_dir = extract_dir / "NHS Hospital Admissions"


if not extract_dir.exists():
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)


def get_year_from_filename(path):
    match = re.search(r"(\d{4}-\d{2})", path.name)
    if match:
        return match.group(1)
    return None


def make_unique_column_names(columns):
    counts = defaultdict(int)
    new_columns = []

    for col in columns:
        counts[col] += 1
        if counts[col] == 1:
            new_columns.append(col)
        else:
            new_columns.append(f"{col}_{counts[col]}")

    return new_columns


def read_primary_diagnosis_summary(path):
    raw = pd.read_excel(
        path,
        sheet_name="Primary Diagnosis Summary",
        header=None
    )

    header_check = raw.apply(
        lambda row: row.astype(str).str.contains(
            "Primary diagnosis: summary code",
            case=False,
            na=False
        ).any(),
        axis=1
    )

    header_row = header_check[header_check].index[0]

    headers = raw.iloc[header_row].tolist()
    headers[0] = "code"
    headers[1] = "description"

    headers = [
        str(h).strip().replace("\n", " ") if pd.notna(h) else f"col_{i}"
        for i, h in enumerate(headers)
    ]

    df = raw.iloc[header_row + 1:].copy()
    df.columns = headers

    df = df.dropna(how="all")
    df = df[df["code"].notna()]
    df = df[df["code"].astype(str).str.lower() != "total"]

    df = df[
        df["code"].astype(str).str.match(
            r"^[A-Z]\d{2}(?:-[A-Z]\d{2})?$",
            na=False
        )
    ]

    rename_map = {
        "Finished consultant episodes": "fce",
        "Admissions": "admissions",
        "Finished Admission Episodes": "admissions",
        "Male": "male",
        "Female": "female",
        "Gender Unknown": "gender_unknown",
        "Emergency": "emergency",
        "Waiting list": "waiting_list",
        "Planned": "planned",
        "Other Admission Method": "other_admission_method",
        "Other": "other_admission_method",
        "Mean time waited": "mean_time_waited",
        "Median time waited": "median_time_waited",
        "Mean length of stay": "mean_length_of_stay",
        "Median length of stay": "median_length_of_stay",
        "Mean age": "mean_age",
    }

    clean_columns = []

    for col in df.columns:
        col = str(col).strip().replace("\n", " ")
        col_without_unit = re.sub(r"\s*\([^)]*\)", "", col).strip()
        clean_columns.append(rename_map.get(col_without_unit, col_without_unit))

    df.columns = make_unique_column_names(clean_columns)

    keep_columns = [
        "code",
        "description",
        "fce",
        "admissions",
        "male",
        "female",
        "gender_unknown",
        "emergency",
        "waiting_list",
        "planned",
        "other_admission_method",
        "mean_time_waited",
        "median_time_waited",
        "mean_length_of_stay",
        "median_length_of_stay",
        "mean_age"
    ]

    keep_columns = [col for col in keep_columns if col in df.columns]
    df = df[keep_columns].copy()

    df["description"] = df["description"].astype(str).str.strip()

    for col in df.columns:
        if col not in ["code", "description"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = get_year_from_filename(path)
    df["source_file"] = path.name

    return df


selected_years = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24"
]

all_data = []

for year in selected_years:
    files = list(data_dir.glob(f"hosp-epis-stat-admi-diag-{year}*.xlsx"))

    if len(files) == 0:
        print(f"No file found for {year}")
        continue

    file_path = files[0]
    print(f"Reading: {file_path.name}")

    year_df = read_primary_diagnosis_summary(file_path)
    all_data.append(year_df)


combined = pd.concat(all_data, ignore_index=True)

combined.to_csv("clean_summary_2018_2024.csv", index=False)

print("Cleaned data saved as clean_summary_2018_2024.csv")
print(combined.head())
print(combined.shape)
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from matplotlib.colors import TwoSlopeNorm

# -----------------------------
# 1. 准备 heatmap 数据
# -----------------------------

df = combined.copy()

pivot = df.pivot_table(
    index=["code", "description"],
    columns="year",
    values="admissions",
    aggfunc="sum"
).reset_index()

base_year = "2019-20"

year_columns = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24"
]

for year in year_columns:
    pivot[year] = pd.to_numeric(pivot[year], errors="coerce")

for year in year_columns:
    pivot[f"{year}_change_pct"] = (
        (pivot[year] - pivot[base_year]) / pivot[base_year] * 100
    ).round(1)

# 只保留 2019-20 年入院人数大于 10000 的类别
important = pivot[pivot[base_year] > 10000].copy()

# 找 2020-21 年变化最大的前 20 个类别
important["abs_lockdown_change"] = important["2020-21_change_pct"].abs()

top_categories = important.sort_values(
    "abs_lockdown_change",
    ascending=False
).head(20)

heatmap_data = top_categories[
    ["code", "description"] + [f"{year}_change_pct" for year in year_columns]
].copy()

heatmap_data.to_csv("heatmap_ready_data.csv", index=False)

print("Heatmap data saved as heatmap_ready_data.csv")
print(heatmap_data)


# -----------------------------
# 2. 画 improved heatmap
# -----------------------------

change_columns = [f"{year}_change_pct" for year in year_columns]

matrix = heatmap_data[change_columns].values

# 颜色显示限制在 -100 到 +100
# 但是格子里的数字仍然显示真实值，比如 +1748%
matrix_for_colour = np.clip(matrix, -100, 100)

labels = [
    code + " " + "\n".join(textwrap.wrap(desc, width=40))
    for code, desc in zip(heatmap_data["code"], heatmap_data["description"])
]

fig, ax = plt.subplots(figsize=(15, 10))

norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)

image = ax.imshow(
    matrix_for_colour,
    aspect="auto",
    cmap="RdBu_r",
    norm=norm
)

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(
    "Percentage change compared with 2019–20 baseline\n"
    "Colour scale clipped at ±100%",
    fontsize=11
)

ax.set_xticks(np.arange(len(year_columns)))
ax.set_xticklabels(year_columns, rotation=45, ha="right")

ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels, fontsize=8)

ax.set_title(
    "Lockdown Fingerprint of Hospital Admissions in England:\n"
    "Percentage Change from 2019–20 Baseline",
    fontsize=15,
    weight="bold"
)

ax.set_xlabel("Financial Year")
ax.set_ylabel("Primary Diagnosis Summary Category")

# 每个格子显示真实百分比
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = matrix[i, j]

        if pd.notna(value):
            ax.text(
                j,
                i,
                f"{value:+.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="black"
            )

# 加白色网格线
ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
ax.tick_params(which="minor", bottom=False, left=False)

# 图下面说明
fig.text(
    0.5,
    0.01,
    "Data level: Primary Diagnosis Summary | Cohort: all ages and all genders | "
    "Baseline: 2019–20 | Values above ±100% are shown as text but clipped in colour.",
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

plt.savefig("lockdown_heatmap_improved.png", dpi=300, bbox_inches="tight")
plt.show()

print("Improved heatmap image saved as lockdown_heatmap_improved.png")
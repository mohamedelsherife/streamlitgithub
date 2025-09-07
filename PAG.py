import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# -------- إعداد الصفحة (اختياري) --------
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# -------- تحميل البيانات -------
@st.cache_data
def load_data():
    return pd.read_csv("Titanic_cleaning_Data.csv")

df_raw = load_data()
df = df_raw.copy()

# ------ متغيرات افتراضية للفلاتر لتفادي UnboundLocalError ------
embarked_sel = "All"

# -------- العنوان --------
st.title("🚢 Titanic Dashboard")

# -------- الفلاتر --------
st.markdown("### Filters")
f1, f2, f3, f4 = st.columns(4)

# فلتر الميناء
with f1:
    if "Embarked" in df.columns:
        embarked_vals = ["All"] + sorted([x for x in df["Embarked"].dropna().unique()])
        embarked_sel = st.selectbox("Embarked (Port)", embarked_vals, index=0)
        if embarked_sel != "All":
            df = df[df["Embarked"] == embarked_sel]

# فلتر الجنس (Male / Female)
with f2:
    if "Sex" in df.columns:
        # لو كانت Sex مُشفّرة كأرقام 0/1 حوّلها إلى نص
        if pd.api.types.is_numeric_dtype(df["Sex"]):
            df["Sex"] = df["Sex"].map({0: "Female", 1: "Male"})
        sexes = ["Male", "Female"]
        sex_sel = st.multiselect("Sex", options=sexes, default=sexes)
        if sex_sel:
            df = df[df["Sex"].isin(sex_sel)]

# فلتر الطبقة
with f3:
    if "Pclass" in df.columns:
        classes = sorted(df["Pclass"].dropna().unique())
        pclass_sel = st.multiselect("Passenger Class (Pclass)", options=classes, default=classes)
        if pclass_sel:
            df = df[df["Pclass"].isin(pclass_sel)]

# فلتر العمر
with f4:
    if "Age" in df.columns:
        if df_raw["Age"].notna().any():
            min_age = int(np.floor(df_raw["Age"].dropna().min()))
            max_age = int(np.ceil(df_raw["Age"].dropna().max()))
        else:
            min_age, max_age = 0, 80
        age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
        df = df[(df["Age"].between(age_range[0], age_range[1])) | (df["Age"].isna())]

st.markdown("---")

# -------- KPIs --------
k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])

total_passengers = len(df)
k1.metric("Number of Passengers", total_passengers)

if "Survived" in df.columns and total_passengers > 0:
    surv_col = pd.to_numeric(df["Survived"], errors="coerce")
    surv_rate = round(surv_col.mean() * 100, 2)
    k2.metric("Survival Rate", f"{surv_rate}%")
else:
    k2.metric("Survival Rate", "-")

if "Age" in df.columns and df["Age"].notna().any():
    k3.metric("Average Age", round(df["Age"].mean(), 1))
else:
    k3.metric("Average Age", "-")

if "Fare" in df.columns and pd.to_numeric(df["Fare"], errors="coerce").notna().any():
    k4.metric("Average Fare", f"${round(pd.to_numeric(df['Fare'], errors='coerce').mean(), 2)}")
else:
    k4.metric("Average Fare", "-")

k5.metric("Embarked Filter", embarked_sel)

st.markdown("---")

# -------- الرسومات --------
c1, c2 = st.columns(2)

# عدد الركاب حسب الطبقة
if "Pclass" in df.columns and len(df) > 0:
    c1.subheader("Number of Passengers by Class")
    by_class = df.groupby("Pclass", observed=True).size().reset_index(name="Count")
    chart_class = (
        alt.Chart(by_class, title="Passengers by Pclass")
        .mark_bar()
        .encode(
            x=alt.X("Pclass:O", title="Pclass"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Pclass:O", scale=alt.Scale(scheme="set2")),
            tooltip=["Pclass", "Count"]
        )
        .properties(height=350)
    )
    c1.altair_chart(chart_class, use_container_width=True)

# معدل النجاة حسب الجنس
if {"Sex", "Survived"}.issubset(df.columns) and len(df) > 0:
    c2.subheader("Average Survival Rate by Sex")
    tmp = df.copy()
    tmp["Survived"] = pd.to_numeric(tmp["Survived"], errors="coerce")
    by_sex = tmp.groupby("Sex", observed=True)["Survived"].mean().dropna().reset_index()
    by_sex["Survival Rate (%)"] = (by_sex["Survived"] * 100).round(2)

    chart_sex = (
        alt.Chart(by_sex, title="Survival Rate by Sex")
        .mark_bar()
        .encode(
            y=alt.Y("Sex:N", title="Sex"),
            x=alt.X("Survival Rate (%):Q", title="Survival Rate (%)"),
            color=alt.Color("Sex:N", scale=alt.Scale(domain=["Male", "Female"], range=["#1f77b4", "#ff69b4"])),
            tooltip=["Sex", "Survival Rate (%)"]
        )
        .properties(height=350)
    )
    c2.altair_chart(chart_sex, use_container_width=True)

c3, c4 = st.columns(2)

# معدل النجاة بالعمر (مجمّع في فئات)
if {"Age", "Survived"}.issubset(df.columns) and df["Age"].notna().any():
    c3.subheader("Survival Rate by Age Group")
    tmp_age = df.dropna(subset=["Age"]).copy()
    tmp_age["Survived"] = pd.to_numeric(tmp_age["Survived"], errors="coerce")
    bins = list(range(0, 81, 5))
    age_binned = pd.cut(tmp_age["Age"], bins=bins, include_lowest=True, right=False)
    surv_by_agebin = (
        tmp_age.groupby(age_binned, observed=True)["Survived"]
        .mean()
        .reset_index(name="Survival Rate (%)")
    )
    surv_by_agebin["Survival Rate (%)"] = (surv_by_agebin["Survival Rate (%)"] * 100).round(2)
    surv_by_agebin["Age Band"] = surv_by_agebin["Age"].astype(str)

    line_age = (
        alt.Chart(surv_by_agebin, title="Survival Rate across Age Bands")
        .mark_line(point=True, color="#ff7f0e")  # برتقالي
        .encode(
            x=alt.X("Age Band:N", title="Age Band (years)"),
            y=alt.Y("Survival Rate (%):Q", title="Survival Rate (%)"),
            tooltip=["Age Band", "Survival Rate (%)"]
        )
        .properties(height=350)
    )
    c3.altair_chart(line_age, use_container_width=True)

# متوسط الأجرة حسب الطبقة
if {"Fare", "Pclass"}.issubset(df.columns) and pd.to_numeric(df["Fare"], errors="coerce").notna().any():
    c4.subheader("Average Fare by Class")
    tmp_fare = df.copy()
    tmp_fare["Fare"] = pd.to_numeric(tmp_fare["Fare"], errors="coerce")
    fare_by_class = tmp_fare.groupby("Pclass", observed=True)["Fare"].mean().reset_index()
    fare_by_class["Average Fare"] = fare_by_class["Fare"].round(2)

    line_fare = (
        alt.Chart(fare_by_class, title="Average Fare by Pclass")
        .mark_line(point=True, color="#2ca02c")  # أخضر
        .encode(
            x=alt.X("Pclass:O", title="Pclass"),
            y=alt.Y("Average Fare:Q", title="Average Fare ($)"),
            tooltip=["Pclass", "Average Fare"]
        )
        .properties(height=350)
    )
    c4.altair_chart(line_fare, use_container_width=True)

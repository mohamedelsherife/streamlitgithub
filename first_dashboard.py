import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime 

# -------- إعداد الصفحة --------
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# -------- دوال التنظيف --------
def remove_duplicated(df):
    if df.duplicated().sum() == 0:
        st.info("✅ No duplicated rows found")
    else:
        df = df.drop_duplicates()
        st.success("🗑️ Duplicates removed")
    return df

def age_missing(df):
    if "Age" in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
        st.success("✅ Missing Age handled")
    return df

def embarked_missing(df):
    if "Embarked" in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        st.success("✅ Missing Embarked handled")
    return df

def Cabin_missing(df):
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])
        st.success("✅ Missing Cabin handled")
    return df

def drop_column(df, col_name):
    if col_name in df.columns:
        df = df.drop(columns=[col_name])
        st.success(f"✅ Column '{col_name}' dropped")
    else:
        st.warning(f"⚠️ Column '{col_name}' not found")
    return df

# -------- الصفحات --------
page = st.sidebar.radio("Navigation", ["Clean Data", "Dashboard"])

# -------- صفحة تنظيف البيانات --------
if page == 'Clean Data':
    st.title("🧹 Data Cleaning")

    # تحميل البيانات بزر
    if st.button('📂 Load Data'):
        st.session_state.df = pd.read_csv('Titanic-Dataset.csv')
        st.success("✅ Data loaded successfully")

    if "df" in st.session_state:
        st.subheader("⚙️ Cleaning Operations")
        pr1, pr2, pr3, pr4, pr5 = st.columns(5)

        with pr1:
            if st.button('🗑️ Drop duplicates'):
                st.session_state.df = remove_duplicated(st.session_state.df)
            st.metric('Number of duplicated', st.session_state.df.duplicated().sum())

        with pr2:
            if st.button('📊 Handle Age missing value'):
                st.session_state.df = age_missing(st.session_state.df)
            st.metric("Missing Age", st.session_state.df['Age'].isnull().sum())

        with pr3:
            if st.button('⚓ Handle Embarked missing value'):
                st.session_state.df = embarked_missing(st.session_state.df) 
            st.metric("Missing Embarked", st.session_state.df['Embarked'].isnull().sum())

        with pr4:
            if st.button('🏠 Remove Cabin column'):
                st.session_state.df = Cabin_missing(st.session_state.df)

        with pr5:
            col_to_drop = st.selectbox("🗂️ Choose a column to drop", st.session_state.df.columns)
            if st.button("❌ Drop Selected Column"):
                st.session_state.df = drop_column(st.session_state.df, col_to_drop)
    
        st.markdown("---")
        st.subheader("📊 Preview of Cleaned Data")
        st.dataframe(st.session_state.df)

        # -------- زر Save Changes --------
        sa1, sa2, sa3 = st.columns([5 ,5, 5])
        with sa1:
            pass
        with sa2:
            if st.button("💾 Save Changes"):
    
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_filename = f"Titanic-Cleaned-{timestamp}.csv"
                st.session_state.df.to_csv(saved_filename, index=False)
                st.session_state["saved_df"] = st.session_state.df.copy()
                st.success(f"✅ Changes saved successfully as '{saved_filename}'!")

        with sa3:
            pass

    else:
        st.warning("⚠️ Please load the data first")

# -------- صفحة الداشبورد --------
elif page == "Dashboard":
    # استخدام النسخة المحفوظة إذا موجودة
    if "saved_df" in st.session_state:
        df_raw = st.session_state["saved_df"]
    else:
        @st.cache_data
        def load_data():
            return pd.read_csv("Titanic-Dataset.csv")
        df_raw = load_data()

    df = df_raw.copy()

    st.title("🚢 Titanic Dashboard")
    st.markdown("### Filters")

    f1, f2, f3, f4 = st.columns(4)

    # فلتر الميناء
    with f1:
        if "Embarked" in df.columns:
            embarked_vals = ["All"] + sorted([x for x in df["Embarked"].dropna().unique()])
            embarked_sel = st.selectbox("Embarked (Port)", embarked_vals, index=0)
            if embarked_sel != "All":
                df = df[df["Embarked"] == embarked_sel]

    # فلتر الجنس
    with f2:
        if "Sex" in df.columns:
            if df["Sex"].dtype in [np.int64, np.float64]:
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
            min_age = int(np.floor(df_raw["Age"].dropna().min())) if df_raw["Age"].notna().any() else 0
            max_age = int(np.ceil(df_raw["Age"].dropna().max())) if df_raw["Age"].notna().any() else 80
            age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
            df = df[(df["Age"].between(age_range[0], age_range[1])) | (df["Age"].isna())]

    st.markdown("---")

    # -------- KPIs --------
    k1, k2, k3, k4, k5 = st.columns([1,1,1,1,1])
    total_passengers = len(df)
    k1.metric("Number of Passengers", total_passengers)

    if "Survived" in df.columns and total_passengers > 0:
        surv_rate = round(df["Survived"].mean() * 100, 2)
        k2.metric("Survival Rate", f"{surv_rate}%")
    else:
        k2.metric("Survival Rate", "-")

    if "Age" in df.columns and df["Age"].notna().any():
        k3.metric("Average Age", round(df["Age"].mean(), 1))
    else:
        k3.metric("Average Age", "-")

    if "Fare" in df.columns and df["Fare"].notna().any():
        k4.metric("Average Fare", f"${round(df['Fare'].mean(), 2)}")
    else:
        k4.metric("Average Fare", "-")

    k5.metric("Embarked Filter", embarked_sel if "embarked_sel" in locals() else "All")

    st.markdown("---")

    # -------- الرسومات --------
    c1, c2 = st.columns(2)

    if {"Pclass"}.issubset(df.columns) and len(df) > 0:
        c1.subheader("Number of Passengers by Class")
        by_class = df.groupby("Pclass").size().reset_index(name="Count")
        chart_class = (
            alt.Chart(by_class, title="Passengers by Pclass")
            .mark_bar()
            .encode(
                x=alt.X("Pclass:O", title="Pclass"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color("Pclass:O", scale=alt.Scale(scheme="set2")),
                tooltip=["Pclass","Count"]
            )
            .properties(height=350)
        )
        c1.altair_chart(chart_class, use_container_width=True)

    if {"Sex","Survived"}.issubset(df.columns) and len(df) > 0:
        c2.subheader("Average Survival Rate by Sex")
        by_sex = df.groupby("Sex")["Survived"].mean().reset_index()
        by_sex["Survival Rate (%)"] = (by_sex["Survived"] * 100).round(2)
        chart_sex = (
            alt.Chart(by_sex, title="Survival Rate by Sex")
            .mark_bar()
            .encode(
                y=alt.Y("Sex:N", title="Sex"),
                x=alt.X("Survival Rate (%):Q", title="Survival Rate (%)"),
                color=alt.Color("Sex:N", scale=alt.Scale(domain=["Male","Female"], range=["#1f77b4","#ff69b4"])),
                tooltip=["Sex","Survival Rate (%)"]
            )
            .properties(height=350)
        )
        c2.altair_chart(chart_sex, use_container_width=True)

    c3, c4 = st.columns(2)

    if {"Age","Survived"}.issubset(df.columns) and df["Age"].notna().any():
        c3.subheader("Survival Rate by Age Group")
        bins = list(range(0,81,5))
        age_binned = pd.cut(df["Age"], bins=bins, include_lowest=True)
        surv_by_agebin = df.groupby(age_binned)["Survived"].mean().reset_index()
        surv_by_agebin["Survival Rate (%)"] = (surv_by_agebin["Survived"] * 100).round(2)
        surv_by_agebin["Age Band"] = surv_by_agebin["Age"].astype(str)

        line_age = (
            alt.Chart(surv_by_agebin, title="Survival Rate across Age Bands")
            .mark_line(point=True, color="#ff7f0e")
            .encode(
                x=alt.X("Age Band:N", title="Age Band (years)"),
                y=alt.Y("Survival Rate (%):Q", title="Survival Rate (%)"),
                tooltip=["Age Band","Survival Rate (%)"]
            )
            .properties(height=350)
        )
        c3.altair_chart(line_age, use_container_width=True)

    if {"Fare","Pclass"}.issubset(df.columns) and df["Fare"].notna().any():
        c4.subheader("Average Fare by Class")
        fare_by_class = df.groupby("Pclass")["Fare"].mean().reset_index()
        fare_by_class["Average Fare"] = fare_by_class["Fare"].round(2)
        line_fare = (
            alt.Chart(fare_by_class, title="Average Fare by Pclass")
            .mark_line(point=True, color="#2ca02c")
            .encode(
                x=alt.X("Pclass:O", title="Pclass"),
                y=alt.Y("Average Fare:Q", title="Average Fare ($)"),
                tooltip=["Pclass","Average Fare"]
            )
            .properties(height=350)
        )
        c4.altair_chart(line_fare, use_container_width=True)

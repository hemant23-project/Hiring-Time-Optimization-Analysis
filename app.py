import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Recruitment Dashboard", layout="wide")

st.title("📊 Recruitment Analytics Dashboard")

# -------------------------
# FILE UPLOAD / DEFAULT LOAD
# -------------------------
st.sidebar.header("📂 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

@st.cache_data
def load_data(file):
    if file is not None:
        # User uploaded file
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    else:
        # Default Excel file
        try:
            df = pd.read_excel("data.xlsx")
            st.sidebar.success("Using default dataset (data.xlsx)")
        except:
            return None

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert date columns
    date_cols = [
        "Application_Date",
        "Screening_Date",
        "Interview_Date",
        "Offer_Date",
        "Joining_Date"
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Create time metrics
    if all(col in df.columns for col in date_cols):
        df["App_to_Screen"] = (df["Screening_Date"] - df["Application_Date"]).dt.days
        df["Screen_to_Interview"] = (df["Interview_Date"] - df["Screening_Date"]).dt.days
        df["Interview_to_Offer"] = (df["Offer_Date"] - df["Interview_Date"]).dt.days
        df["Offer_to_Join"] = (df["Joining_Date"] - df["Offer_Date"]).dt.days
        df["Total_Hiring_Time"] = (df["Joining_Date"] - df["Application_Date"]).dt.days

    return df

df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ Please upload a dataset or place 'data.xlsx' in the app folder.")
    st.stop()

# -------------------------
# DATA PREVIEW
# -------------------------
st.subheader("👀 Dataset Preview")
st.dataframe(df.head())

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("🔍 Filters")

job_roles = st.sidebar.multiselect(
    "Job Role",
    df["Job_Role"].dropna().unique(),
    default=df["Job_Role"].dropna().unique()
)

sources = st.sidebar.multiselect(
    "Source",
    df["Source"].dropna().unique(),
    default=df["Source"].dropna().unique()
)

recruiters = st.sidebar.multiselect(
    "Recruiter",
    df["Recruiter_ID"].dropna().unique(),
    default=df["Recruiter_ID"].dropna().unique()
)

date_range = st.sidebar.date_input(
    "Application Date Range",
    [df["Application_Date"].min(), df["Application_Date"].max()]
)

# Apply filters
filtered_df = df[
    (df["Job_Role"].isin(job_roles)) &
    (df["Source"].isin(sources)) &
    (df["Recruiter_ID"].isin(recruiters)) &
    (df["Application_Date"] >= pd.to_datetime(date_range[0])) &
    (df["Application_Date"] <= pd.to_datetime(date_range[1]))
]

# -------------------------
# KPI METRICS
# -------------------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Candidates", len(filtered_df))

if "Total_Hiring_Time" in filtered_df.columns:
    col2.metric("Avg Hiring Time", round(filtered_df["Total_Hiring_Time"].mean(), 2))

if "Offer_to_Join" in filtered_df.columns:
    col3.metric("Offer → Join Time", round(filtered_df["Offer_to_Join"].mean(), 2))

# -------------------------
# VISUALS
# -------------------------
st.subheader("📈 Visual Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("### Hiring Time by Role")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="Job_Role", y="Total_Hiring_Time", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.write("### Source Distribution")
    fig, ax = plt.subplots()
    filtered_df["Source"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)

# -------------------------
# FUNNEL ANALYSIS
# -------------------------
st.subheader("⏳ Hiring Funnel")

stage_cols = [
    "App_to_Screen",
    "Screen_to_Interview",
    "Interview_to_Offer",
    "Offer_to_Join"
]

if all(col in filtered_df.columns for col in stage_cols):
    st.bar_chart(filtered_df[stage_cols].mean())

# -------------------------
# PREDICTION
# -------------------------
st.subheader("🤖 Hiring Time Prediction")

model_df = filtered_df.dropna(subset=["Total_Hiring_Time"])

if len(model_df) > 2:
    X = pd.get_dummies(model_df[["Job_Role", "Source", "Recruiter_ID"]])
    y = model_df["Total_Hiring_Time"]

    model = LinearRegression()
    model.fit(X, y)

    role_input = st.selectbox("Job Role", df["Job_Role"].unique())
    source_input = st.selectbox("Source", df["Source"].unique())
    recruiter_input = st.selectbox("Recruiter", df["Recruiter_ID"].unique())

    input_df = pd.DataFrame({
        "Job_Role": [role_input],
        "Source": [source_input],
        "Recruiter_ID": [recruiter_input]
    })

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_encoded)

    st.success(f"Estimated Hiring Time: {round(prediction[0], 2)} days")

else:
    st.info("Not enough data for prediction")

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("🧠 Insights")

if "Total_Hiring_Time" in filtered_df.columns:
    avg_time = round(filtered_df["Total_Hiring_Time"].mean(), 1)
    top_source = filtered_df["Source"].value_counts().idxmax()

    st.write(f"""
    - 📌 Average hiring time: **{avg_time} days**
    - 🎯 Top source: **{top_source}**
    """)

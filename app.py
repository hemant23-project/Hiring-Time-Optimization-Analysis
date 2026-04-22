import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

st.title("📊 Hiring Time Optimization Dashboard")

# File Upload
file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file is not None:
    df = pd.read_excel(file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Convert dates
    df['Application_Date'] = pd.to_datetime(df['Application_Date'])
    df['Screening_Date'] = pd.to_datetime(df['Screening_Date'])
    df['Interview_Date'] = pd.to_datetime(df['Interview_Date'])
    df['Offer_Date'] = pd.to_datetime(df['Offer_Date'])
    df['Joining_Date'] = pd.to_datetime(df['Joining_Date'])

    # KPIs
    total = len(df)
    screened = df['Screening_Date'].notna().sum()
    interviewed = df['Interview_Date'].notna().sum()
    offered = df['Offer_Date'].notna().sum()
    joined = df['Joining_Date'].notna().sum()

    st.write("## 📌 Hiring Funnel KPIs")
    st.write(f"Total Applications: {total}")
    st.write(f"Screened: {screened}")
    st.write(f"Interviewed: {interviewed}")
    st.write(f"Offered: {offered}")
    st.write(f"Joined: {joined}")

    # Selected Candidates
    selected_df = df[df['Status'] == 'Selected'].copy()

    # Time Columns
    selected_df['Time_to_Screen'] = (selected_df['Screening_Date'] - selected_df['Application_Date']).dt.days
    selected_df['Time_to_Interview'] = (selected_df['Interview_Date'] - selected_df['Screening_Date']).dt.days
    selected_df['Time_to_Offer'] = (selected_df['Offer_Date'] - selected_df['Interview_Date']).dt.days
    selected_df['Time_to_Join'] = (selected_df['Joining_Date'] - selected_df['Offer_Date']).dt.days

    selected_df['Total_Hiring_Time'] = (selected_df['Joining_Date'] - selected_df['Application_Date']).dt.days

    st.write("## ⏱ Hiring Time Analysis")
    st.write("Average Hiring Time:", selected_df['Total_Hiring_Time'].mean())

    stage_avg = selected_df[['Time_to_Screen','Time_to_Interview','Time_to_Offer','Time_to_Join']].mean()
    st.write("Stage-wise Time:", stage_avg)

    st.write("Bottleneck Stage:", stage_avg.idxmax())

    # 📊 Charts
    st.write("## 📊 Visualizations")

    fig1, ax1 = plt.subplots()
    sns.countplot(x='Status', data=df, ax=ax1)
    ax1.set_title("Hiring Funnel")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.histplot(selected_df['Total_Hiring_Time'], kde=True, ax=ax2)
    ax2.set_title("Hiring Time Distribution")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Job_Role', y='Total_Hiring_Time', data=selected_df, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.barplot(x='Source', y='Total_Hiring_Time', data=selected_df, ax=ax4)
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    # Trend
    selected_df['Month'] = selected_df['Application_Date'].dt.month

    fig5, ax5 = plt.subplots()
    sns.lineplot(x='Month', y='Total_Hiring_Time', data=selected_df, ax=ax5)
    st.pyplot(fig5)

    # ML
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score

    ml_df = selected_df.dropna()

    X = ml_df[['Time_to_Screen','Time_to_Interview','Time_to_Offer','Time_to_Join']]
    y = ml_df['Total_Hiring_Time']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("## 🤖 Model Performance")
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))
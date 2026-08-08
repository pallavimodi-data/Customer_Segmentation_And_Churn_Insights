
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="chart",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")

    # Drop unnecessary columns
    drop_cols = ["RowNumber", "CustomerId", "Surname"]

    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df


df = load_data()

# ---------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------

# Age Group
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18, 30, 45, 60, 100],
    labels=["18-30", "31-45", "46-60", "60+"]
)

# Balance Segment
df["BalanceSegment"] = pd.cut(
    df["Balance"],
    bins=[-1, 0, 100000, 300000],
    labels=["Zero Balance", "Medium Balance", "High Balance"]
)

# Tenure Group
df["TenureGroup"] = pd.cut(
    df["Tenure"],
    bins=[-1, 3, 7, 10],
    labels=["New", "Mid-Term", "Loyal"]
)

# Churn Label
df["ChurnStatus"] = df["Exited"].map({
    0: "Retained",
    1: "Churned"
})

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("Customer Filters")

selected_geo = st.sidebar.multiselect(
    "Select Geography",
    options=df["Geography"].unique(),
    default=df["Geography"].unique()
)

selected_gender = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

age_options = df["AgeGroup"].dropna().astype(str).unique().tolist()

selected_age = st.sidebar.multiselect(
    "Select Age Group",
    options=age_options,
    default=age_options
)

tenure_options = df["TenureGroup"].dropna().astype(str).unique().tolist()

selected_tenure = st.sidebar.multiselect(
    "Select Tenure Group",
    options=tenure_options,
    default=tenure_options
)
# ---------------------------------------------------
# FILTER DATA
# ---------------------------------------------------
filtered_df = df[
    (df["Geography"].isin(selected_geo)) &
    (df["Gender"].isin(selected_gender)) &
    (df["AgeGroup"].isin(selected_age)) &
    (df["TenureGroup"].isin(selected_tenure))
]

# ---------------------------------------------------
# DASHBOARD TITLE
# ---------------------------------------------------
st.title("Customer Segmentation & Churn Insights Dashboard")

st.markdown("This interactive dashboard provides customer churn insights, customer segmentation analysis,and high-value customer behavior tracking using Streamlit and Plotly.")

st.markdown("---")

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------
total_customers = filtered_df.shape[0]
total_churned = filtered_df["Exited"].sum()
churn_rate = round((total_churned / total_customers) * 100, 2)

avg_balance = round(filtered_df["Balance"].mean(), 2)
avg_salary = round(filtered_df["EstimatedSalary"].mean(), 2)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{total_churned:,}")
col3.metric("Churn Rate", f"{churn_rate}%")
col4.metric("Avg Balance", f"${avg_balance:,.0f}")
col5.metric("Avg Salary", f"${avg_salary:,.0f}")

st.markdown("---")

# ---------------------------------------------------
# OVERALL CHURN SUMMARY
# ---------------------------------------------------
st.subheader("Overall Churn Summary")

col6, col7 = st.columns(2)

with col6:
    churn_dist = filtered_df["ChurnStatus"].value_counts().reset_index()
    churn_dist.columns = ["Status", "Count"]

    fig_pie = px.pie(
        churn_dist,
        names="Status",
        values="Count",
        hole=0.5,
        title="Customer Churn Distribution"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

with col7:
    churn_geo = (
        filtered_df.groupby("Geography")["Exited"]
        .mean()
        .reset_index()
    )

    churn_geo["Exited"] = churn_geo["Exited"] * 100

    fig_geo = px.bar(
        churn_geo,
        x="Geography",
        y="Exited",
        text_auto=".2f",
        title="Geography-wise Churn Rate",
        labels={"Exited": "Churn Rate (%)"}
    )

    st.plotly_chart(fig_geo, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# AGE & TENURE ANALYSIS
# ---------------------------------------------------
st.subheader("Age & Tenure Churn Comparison")

col8, col9 = st.columns(2)

with col8:
    fig_age = px.box(
        filtered_df,
        x="ChurnStatus",
        y="Age",
        color="ChurnStatus",
        title="Age Distribution by Churn Status"
    )

    st.plotly_chart(fig_age, use_container_width=True)

with col9:
    fig_tenure = px.box(
        filtered_df,
        x="ChurnStatus",
        y="Tenure",
        color="ChurnStatus",
        title="Tenure Distribution by Churn Status"
    )

    st.plotly_chart(fig_tenure, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# HIGH VALUE CUSTOMER EXPLORER
# ---------------------------------------------------
st.subheader("High-Value Customer Churn Explorer")

balance_threshold = st.slider(
    "Select Minimum Balance",
    min_value=0,
    max_value=int(df["Balance"].max()),
    value=100000
)

high_value_df = filtered_df[
    filtered_df["Balance"] >= balance_threshold
]

fig_scatter = px.scatter(
    high_value_df,
    x="Balance",
    y="EstimatedSalary",
    color="ChurnStatus",
    size="CreditScore",
    hover_data=["Geography", "Gender", "Age"],
    title="High-Value Customer Churn Analysis"
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# CUSTOMER SEGMENT ANALYSIS
# ---------------------------------------------------
st.subheader("Customer Segment Analysis")

segment_churn = (
    filtered_df.groupby("AgeGroup")["Exited"]
    .mean()
    .reset_index()
)

segment_churn["Exited"] = segment_churn["Exited"] * 100

fig_segment = px.line(
    segment_churn,
    x="AgeGroup",
    y="Exited",
    markers=True,
    title="Churn Rate Across Age Groups",
    labels={"Exited": "Churn Rate (%)"}
)

st.plotly_chart(fig_segment, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# DRILL DOWN VIEW
# ---------------------------------------------------
st.subheader("Customer Drill-Down View")

selected_country = st.selectbox(
    "Select Geography",
    filtered_df["Geography"].unique()
)

drill_df = filtered_df[
    filtered_df["Geography"] == selected_country
]

st.dataframe(
    drill_df[
        [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "ChurnStatus"
        ]
    ],
    use_container_width=True
)

# ---------------------------------------------------
# DOWNLOAD SECTION
# ---------------------------------------------------
st.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_customer_data.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.caption(
    "Built with Streamlit | Plotly | Pandas | Machine Learning Analytics"
)


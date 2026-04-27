import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FMCG Demand Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { padding: 1rem; border-radius: 10px; }
    .stMetric label { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD & CLEAN DATA ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_Name'] = df['Date'].dt.strftime('%b %Y')
    df['Quarter'] = df['Date'].dt.quarter
    df['Is_Holiday'] = df['Holiday_Flag'].map({1: 'Holiday Week', 0: 'Regular Week'})
    df['Store'] = df['Store'].astype(str)
    # Assign FMCG-style categories to stores
    categories = ['Home Care', 'Beauty & Grooming', 'Health & Wellness',
                  'Food & Beverage', 'Baby Care', 'Fabric Care',
                  'Personal Care', 'Oral Care', 'Hair Care']
    store_ids = sorted(df['Store'].unique(), key=lambda x: int(x))
    cat_map = {s: categories[i % len(categories)] for i, s in enumerate(store_ids)}
    df['Category'] = df['Store'].map(cat_map)
    return df

df = load_data()

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 Dashboard Controls")
st.sidebar.title("🔍 Filters")

all_stores = sorted(df['Store'].unique(), key=lambda x: int(x))
selected_stores = st.sidebar.multiselect("Select Stores", all_stores, default=all_stores[:10])

all_years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect("Select Year(s)", all_years, default=all_years)

holiday_filter = st.sidebar.radio("Week Type", ["All", "Holiday Weeks", "Regular Weeks"])

# Apply filters
filtered = df[df['Store'].isin(selected_stores) & df['Year'].isin(selected_years)]
if holiday_filter == "Holiday Weeks":
    filtered = filtered[filtered['Holiday_Flag'] == 1]
elif holiday_filter == "Regular Weeks":
    filtered = filtered[filtered['Holiday_Flag'] == 0]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📊 FMCG Consumer Demand Analytics Dashboard")
st.markdown("**Retail sales intelligence platform** — tracking demand trends, store performance, and forecasting future sales across product categories.")
st.markdown("---")

# ── KPI METRICS ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total_sales = filtered['Weekly_Sales'].sum()
avg_weekly = filtered['Weekly_Sales'].mean()
best_store = filtered.groupby('Store')['Weekly_Sales'].sum().idxmax()
holiday_lift = (
    filtered[filtered['Holiday_Flag']==1]['Weekly_Sales'].mean() /
    filtered[filtered['Holiday_Flag']==0]['Weekly_Sales'].mean() - 1
) * 100 if len(filtered[filtered['Holiday_Flag']==0]) > 0 else 0

col1.metric("💰 Total Revenue", f"${total_sales/1e9:.2f}B")
col2.metric("📦 Avg Weekly Sales", f"${avg_weekly/1e6:.2f}M")
col3.metric("🏆 Best Performing Store", f"Store {best_store}")
col4.metric("🎄 Holiday Sales Lift", f"+{holiday_lift:.1f}%")

st.markdown("---")

# ── ROW 1: SALES TREND + STORE PERFORMANCE ───────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Weekly Sales Trend")
    trend = filtered.groupby('Date')['Weekly_Sales'].sum().reset_index()
    fig_trend = px.line(
        trend, x='Date', y='Weekly_Sales',
        labels={'Weekly_Sales': 'Total Weekly Sales ($)', 'Date': ''},
        color_discrete_sequence=['#003087']
    )
    fig_trend.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis_tickformat='$,.0f', height=320,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    st.subheader("🏪 Top 10 Stores by Revenue")
    store_sales = filtered.groupby('Store')['Weekly_Sales'].sum().nlargest(10).reset_index()
    fig_stores = px.bar(
        store_sales, x='Weekly_Sales', y='Store',
        orientation='h',
        color='Weekly_Sales',
        color_continuous_scale=['#cce0ff', '#003087'],
        labels={'Weekly_Sales': 'Total Sales ($)', 'Store': 'Store ID'}
    )
    fig_stores.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickformat='$,.0f', height=320,
        margin=dict(t=20, b=20), showlegend=False,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_stores, use_container_width=True)

# ── ROW 2: CATEGORY BREAKDOWN + HOLIDAY IMPACT ───────────────────────────────
col_left2, col_right2 = st.columns([1, 1])

with col_left2:
    st.subheader("🛒 Sales by FMCG Category")
    cat_sales = filtered.groupby('Category')['Weekly_Sales'].sum().reset_index()
    fig_cat = px.pie(
        cat_sales, values='Weekly_Sales', names='Category',
        color_discrete_sequence=px.colors.sequential.Blues_r,
        hole=0.4
    )
    fig_cat.update_layout(height=340, margin=dict(t=20, b=20))
    st.plotly_chart(fig_cat, use_container_width=True)

with col_right2:
    st.subheader("🎄 Holiday vs Regular Week Sales")
    holiday_sales = filtered.groupby(['Is_Holiday'])['Weekly_Sales'].mean().reset_index()
    fig_hol = px.bar(
        holiday_sales, x='Is_Holiday', y='Weekly_Sales',
        color='Is_Holiday',
        color_discrete_map={'Holiday Week': '#003087', 'Regular Week': '#90b8e8'},
        labels={'Weekly_Sales': 'Avg Weekly Sales ($)', 'Is_Holiday': ''},
        text_auto='.3s'
    )
    fig_hol.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis_tickformat='$,.0f', height=340,
        margin=dict(t=20, b=20), showlegend=False
    )
    st.plotly_chart(fig_hol, use_container_width=True)

# ── ROW 3: ECONOMIC FACTORS ───────────────────────────────────────────────────
st.subheader("🌡️ Economic Factors vs Sales Performance")
col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    fig_temp = px.scatter(
        filtered.sample(min(500, len(filtered))),
        x='Temperature', y='Weekly_Sales',
        color='Is_Holiday',
        color_discrete_map={'Holiday Week': '#003087', 'Regular Week': '#90b8e8'},
        labels={'Weekly_Sales': 'Weekly Sales ($)', 'Temperature': 'Temperature (°F)'},
        title="Temperature vs Sales",
        trendline='ols'
    )
    fig_temp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=40, b=20), showlegend=False)
    st.plotly_chart(fig_temp, use_container_width=True)

with col_e2:
    fig_fuel = px.scatter(
        filtered.sample(min(500, len(filtered))),
        x='Fuel_Price', y='Weekly_Sales',
        color='Is_Holiday',
        color_discrete_map={'Holiday Week': '#003087', 'Regular Week': '#90b8e8'},
        labels={'Weekly_Sales': 'Weekly Sales ($)', 'Fuel_Price': 'Fuel Price ($)'},
        title="Fuel Price vs Sales",
        trendline='ols'
    )
    fig_fuel.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=40, b=20), showlegend=False)
    st.plotly_chart(fig_fuel, use_container_width=True)

with col_e3:
    fig_unemp = px.scatter(
        filtered.sample(min(500, len(filtered))),
        x='Unemployment', y='Weekly_Sales',
        color='Is_Holiday',
        color_discrete_map={'Holiday Week': '#003087', 'Regular Week': '#90b8e8'},
        labels={'Weekly_Sales': 'Weekly Sales ($)', 'Unemployment': 'Unemployment Rate (%)'},
        title="Unemployment vs Sales",
        trendline='ols'
    )
    fig_unemp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            height=280, margin=dict(t=40, b=20), showlegend=False)
    st.plotly_chart(fig_unemp, use_container_width=True)

# ── ROW 4: ML FORECASTING ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Demand Forecasting Model — Linear Regression")

features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag', 'Month', 'Quarter']
ml_data = df[features + ['Weekly_Sales']].dropna()

X = ml_data[features]
y = ml_data['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("📐 Model R² Score", f"{r2:.3f}", help="Closer to 1.0 = better fit")
col_m2.metric("📉 Mean Abs Error", f"${mae:,.0f}", help="Average prediction error per week")
col_m3.metric("🔢 Training Samples", f"{len(X_train):,}")

# Actual vs Predicted chart
col_chart, col_imp = st.columns([2, 1])
with col_chart:
    sample_idx = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
    actual_sample = y_test.iloc[sample_idx].values
    pred_sample = y_pred[sample_idx]

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=list(range(len(actual_sample))), y=actual_sample,
        mode='lines', name='Actual Sales', line=dict(color='#003087', width=2)
    ))
    fig_pred.add_trace(go.Scatter(
        x=list(range(len(pred_sample))), y=pred_sample,
        mode='lines', name='Predicted Sales',
        line=dict(color='#ff6b35', width=2, dash='dash')
    ))
    fig_pred.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis_tickformat='$,.0f', height=300,
        margin=dict(t=20, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis_title='Sample Index', yaxis_title='Weekly Sales ($)'
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with col_imp:
    st.markdown("#### 📌 Feature Importance")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Impact': np.abs(model.coef_)
    }).sort_values('Impact', ascending=True)

    fig_coef = px.bar(
        coef_df, x='Impact', y='Feature', orientation='h',
        color='Impact', color_continuous_scale=['#cce0ff', '#003087']
    )
    fig_coef.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(t=10, b=10),
        coloraxis_showscale=False, showlegend=False
    )
    st.plotly_chart(fig_coef, use_container_width=True)

# ── INTERACTIVE PREDICTOR ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Predict Weekly Sales — Interactive Tool")
st.markdown("Adjust the parameters below to predict demand for any scenario:")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    p_temp = st.slider("Temperature (°F)", 10, 110, 65)
    p_holiday = st.selectbox("Week Type", [0, 1], format_func=lambda x: "Holiday" if x else "Regular")
with col_p2:
    p_fuel = st.slider("Fuel Price ($)", 2.0, 5.0, 3.5, 0.1)
    p_month = st.slider("Month", 1, 12, 6)
with col_p3:
    p_cpi = st.slider("CPI", 120.0, 230.0, 180.0, 0.5)
    p_quarter = st.selectbox("Quarter", [1, 2, 3, 4])
with col_p4:
    p_unemp = st.slider("Unemployment (%)", 3.0, 15.0, 7.0, 0.1)

input_data = np.array([[p_temp, p_fuel, p_cpi, p_unemp, p_holiday, p_month, p_quarter]])
prediction = model.predict(input_data)[0]

st.markdown(f"""
<div style='background: linear-gradient(135deg, #003087, #0057b8); color: white;
            padding: 1.5rem; border-radius: 12px; text-align: center; margin-top: 1rem;'>
    <h3 style='color: white; margin: 0;'>📦 Predicted Weekly Sales</h3>
    <h1 style='color: #ffd700; margin: 0.5rem 0;'>${prediction:,.2f}</h1>
    <p style='color: #cce0ff; margin: 0;'>Based on selected economic & seasonal parameters</p>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    📊 FMCG Demand Analytics Dashboard &nbsp;|&nbsp; Built with Python, Streamlit & Scikit-learn &nbsp;|&nbsp; Ridhima Bhatia
</div>
""", unsafe_allow_html=True)
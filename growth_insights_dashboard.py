
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Growth Insights Dashboard", layout="wide")
st.title("📊 Growth Insights Dashboard for Retail Analytics")

uploaded_file = st.file_uploader("📤 Upload your dataset (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
else:
    st.warning("⚠️ Please upload the 'DATASET FOR ANUPAM.xlsx' file to continue.")
    st.stop()

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["📈 Regression", "📅 Forecasting", "🧪 Causal Inference", "🔗 Basket Recommendations"])

if section == "📈 Regression":
    st.header("🔮 Predictive Modeling: Category Sales Prediction")
    df_model = df.dropna(subset=['category_sales_value', 'basket_spend', 'product_unit', 'category_level1'])
    df_model = pd.get_dummies(df_model, columns=['category_level1'], drop_first=True)
    X = df_model.drop(columns=['category_sales_value'])
    y = df_model['category_sales_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': "Actual", 'y': "Predicted"}, title="Actual vs Predicted Sales")
    st.plotly_chart(fig)
    st.markdown("✅ **Client Insight:**")
    st.markdown(f"- R²: **{r2:.2f}**, MAE: **£{mae:,.2f}**. Indicates prediction reliability.")
    coeffs = pd.Series(model.coef_, index=X.columns).sort_values()
    fig2 = px.bar(coeffs, title="Feature Importance")
    st.plotly_chart(fig2)

if section == "📅 Forecasting":
    st.header("📅 Time Series Forecasting per Category")
    if 'year_of_date' in df.columns and 'month_of_year' in df.columns:
        try:
            df['year_of_date'] = pd.to_numeric(df['year_of_date'], errors='coerce')
            df['month_of_year'] = pd.to_numeric(df['month_of_year'], errors='coerce')
            df.dropna(subset=['year_of_date', 'month_of_year'], inplace=True)
            df['year_of_date'] = df['year_of_date'].astype(int)
            df['month_of_year'] = df['month_of_year'].astype(int)
            df['date'] = pd.to_datetime(dict(year=df['year_of_date'], month=df['month_of_year'], day=1))
            category_options = df['category_level1'].dropna().unique()
            selected_cat = st.selectbox("Select a Category", sorted(category_options))
            df_cat = df[df['category_level1'] == selected_cat]
            df_cat_monthly = df_cat.groupby('date').agg({'category_sales_value': 'sum'}).reset_index()
            df_cat_monthly.columns = ['ds', 'y']
            if len(df_cat_monthly) >= 6:
                from prophet import Prophet
                model = Prophet()
                model.fit(df_cat_monthly)
                future = model.make_future_dataframe(periods=3, freq='MS')
                forecast = model.predict(future)
                fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"Sales Forecast: {selected_cat}")
                fig_forecast.add_scatter(x=df_cat_monthly['ds'], y=df_cat_monthly['y'], mode='markers', name='Actual')
                st.plotly_chart(fig_forecast)
                st.markdown("✅ **Client Insight:** Use forecast for inventory and staffing plans.")
            else:
                st.warning("Not enough monthly data (min 6 months).")
        except Exception as e:
            st.error(f"Error during forecasting: {e}")

if section == "🧪 Causal Inference":
    st.header("🧪 Causal Inference: Symbol Effectiveness")
    if 'symbol_indicator' in df.columns:
        df_causal = df[['category_sales_value', 'symbol_indicator']].dropna()
        df_causal['symbol_indicator'] = df_causal['symbol_indicator'].astype(int)
        treated = df_causal[df_causal['symbol_indicator'] == 1]
        control = df_causal[df_causal['symbol_indicator'] == 0]
        uplift = treated['category_sales_value'].mean() - control['category_sales_value'].mean()
        st.metric("Symbol", f"£{treated['category_sales_value'].mean():,.2f}")
        st.metric("No Symbol", f"£{control['category_sales_value'].mean():,.2f}")
        st.metric("Uplift", f"£{uplift:,.2f}")
        st.markdown("✅ **Client Insight:** Marketing symbols appear to increase category sales.")
    else:
        st.warning("Column 'symbol_indicator' missing.")

if section == "🔗 Basket Recommendations":
    st.header("🔗 Basket-Based Recommendations")
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    if 'basket_id' in df.columns and 'category_level1' in df.columns:
        baskets = df[['basket_id', 'category_level1']].dropna().groupby('basket_id')['category_level1'].apply(list).tolist()
        te = TransactionEncoder()
        basket_matrix = te.fit_transform(baskets)
        df_matrix = pd.DataFrame(basket_matrix, columns=te.columns_)
        freq_items = apriori(df_matrix, min_support=0.01, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1)
        if not rules.empty:
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))
            st.markdown("✅ **Client Insight:** Promote product combinations frequently bought together.")
        else:
            st.warning("No frequent itemsets found.")
    else:
        st.warning("Missing required columns: 'basket_id' and/or 'category_level1'")

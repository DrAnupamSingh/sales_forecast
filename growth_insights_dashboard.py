
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Growth Insights Dashboard", layout="wide")
st.title("📊 Growth Insights Dashboard for Retail Analytics")

df = pd.read_excel("DATASET FOR ANUPAM.xlsx")

# Sidebar Navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["📈 Regression", "📅 Forecasting", "🧪 Causal Inference", "🔗 Basket Recommendations"])

if section == "📈 Regression":
    st.header("🔮 Predictive Modeling: Category Sales Prediction")

    df_model = df.dropna(subset=['category_sales_value'])
    
    # One-hot encode categorical columns
    cat_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    
    # Separate features and target
    X = df_model.drop(columns=['category_sales_value'])
    y = df_model['category_sales_value']

    # Ensure all inputs are numeric and aligned
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    df_clean = pd.concat([X, y], axis=1).dropna()
    X = df_clean.drop(columns=['category_sales_value'])
    y = df_clean['category_sales_value']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Plot predictions
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': "Actual", 'y': "Predicted"}, title="Actual vs Predicted Sales")
    st.plotly_chart(fig)

    st.markdown("✅ **Client Insight:**")
    st.markdown(f"- R² Score: **{r2:.2f}**")
    st.markdown(f"- Mean Absolute Error (MAE): **£{mae:,.2f}**")

    # Feature importance
    coeffs = pd.Series(model.coef_, index=X.columns).sort_values()
    fig2 = px.bar(coeffs, title="Feature Importance")
    st.plotly_chart(fig2)



# ================================
# 📅 TIME SERIES FORECASTING MODULE
# ================================
if section == "📅 Forecasting":
    st.header("📅 Time Series Forecasting per Category")

    if 'year_of_date' in df.columns and 'month_of_year' in df.columns:
        try:
            df['year_of_date'] = pd.to_numeric(df['year_of_date'], errors='coerce')
            df['month_of_year'] = pd.to_numeric(df['month_of_year'], errors='coerce')
            df = df.dropna(subset=['year_of_date', 'month_of_year'])
            df['year_of_date'] = df['year_of_date'].astype(int)
            df['month_of_year'] = df['month_of_year'].astype(int)
            df['date'] = pd.to_datetime(dict(year=df['year_of_date'], month=df['month_of_year'], day=1))

            category_options = df['category_level1'].dropna().unique()
            selected_cat = st.selectbox("Select a Category to Forecast", sorted(category_options))

            df_cat = df[df['category_level1'] == selected_cat]
            df_cat_monthly = df_cat.groupby('date').agg({'category_sales_value': 'sum'}).reset_index()
            df_cat_monthly.columns = ['ds', 'y']

            if len(df_cat_monthly) >= 6:
                from prophet import Prophet
                model = Prophet()
                model.fit(df_cat_monthly)
                future = model.make_future_dataframe(periods=3, freq='MS')
                forecast = model.predict(future)

                fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"Sales Forecast for {selected_cat}")
                fig_forecast.add_scatter(x=df_cat_monthly['ds'], y=df_cat_monthly['y'], mode='markers', name='Actual')
                st.plotly_chart(fig_forecast)

                st.markdown("✅ **Client Insight:**")
                st.markdown("- Use forecast for monthly procurement and staffing.")
                st.markdown("- Helps prepare for peak or low seasons.")
            else:
                st.warning("Not enough data to forecast this category (minimum 6 months required).")
        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
    else:
        st.error("Missing 'year_of_date' and/or 'month_of_year' columns.")


# ================================
# 🔗 BASKET-BASED RECOMMENDATIONS
# ================================
if section == "🔗 Basket Recommendations":
    st.header("🔗 Basket-Based Recommendations")

    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    if 'basket_id' in df.columns and 'category_level1' in df.columns:
        transactions = df[['basket_id', 'category_level1']].dropna().groupby('basket_id')['category_level1'].apply(list).tolist()
        te = TransactionEncoder()
        matrix = te.fit_transform(transactions)
        df_matrix = pd.DataFrame(matrix, columns=te.columns_)

        freq_items = apriori(df_matrix, min_support=0.01, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1)

        if not rules.empty:
            top_rules = rules.sort_values(by='lift', ascending=False).head(10)
            st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

            st.markdown("✅ **Client Insight:**")
            st.markdown("- Cross-sell products often bought together.")
            st.markdown("- Build bundle offers based on these pairings.")
        else:
            st.warning("No strong basket rules found. Try more data or lower thresholds.")
    else:
        st.warning("Required columns: 'basket_id', 'category_level1'")

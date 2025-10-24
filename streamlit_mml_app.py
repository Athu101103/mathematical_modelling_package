# -*- coding: utf-8 -*-
"""
Comprehensive Mathematical Modeling in Finance - Streamlit App
Interactive implementation of ARIMA, Portfolio Optimization, CAPM, and SIM models
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from scipy import optimize
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Mathematical Modeling in Finance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'returns_data' not in st.session_state:
    st.session_state.returns_data = None
if 'company_returns' not in st.session_state:
    st.session_state.company_returns = {}

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Module",
    [
        "🏠 Home & Data Loading",
        "📈 Time Series Analysis",
        "🔍 ARIMA Modeling", 
        "💼 Portfolio Optimization",
        "📊 CAPM Analysis",
        "🎯 Simple Index Model",
        "⚖️ Model Comparison"
    ]
)

# ===============================================================
# UTILITY FUNCTIONS (from original code)
# ===============================================================

@st.cache_data
def download_crypto_data(tickers, start_date="2017-01-01"):
    """Download and preprocess cryptocurrency data"""
    all_data = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, progress=False)
            if data.empty:
                st.warning(f"⚠️ No data found for {ticker}")
                continue
                
            data['logp'] = np.log(data['Close'])
            data['r'] = data['logp'].diff()
            data['log_vol'] = np.log(data['Volume'] + 1)
            data = data.dropna()
            data['Ticker'] = ticker
            data = data[['r', 'log_vol', 'Ticker']]
            all_data.append(data)
            
        except Exception as e:
            st.error(f"Error downloading {ticker}: {str(e)}")
            continue
    
    if not all_data:
        return None, None, None
        
    combined_data = pd.concat(all_data).sort_index()
    returns_matrix = combined_data.pivot_table(
        index=combined_data.index, 
        columns='Ticker', 
        values='r'
    )
    
    # Ensure column names are clean strings
    returns_matrix.columns = [str(col).strip() for col in returns_matrix.columns]
    returns_matrix = returns_matrix.loc[:, returns_matrix.columns != '']  # Remove empty columns
    
    # Remove any columns that might have become tuples or other non-string types
    valid_columns = []
    for col in returns_matrix.columns:
        if isinstance(col, (tuple, list)):
            # If it's a tuple/list, take the last non-empty element
            col_str = str([x for x in col if x][-1]) if col else "Unknown"
        else:
            col_str = str(col)
        valid_columns.append(col_str)
    
    returns_matrix.columns = valid_columns
    
    company_returns = {
        ticker: returns_matrix[ticker].dropna().values 
        for ticker in returns_matrix.columns
    }
    
    return combined_data, returns_matrix, company_returns

def calculate_acf(series, max_lags):
    """Calculate Autocorrelation Function"""
    N = len(series)
    demeaned_series = series - np.mean(series)
    c0 = np.sum(demeaned_series**2) / N
    acf_values = []
    
    for k in range(max_lags + 1):
        if k == 0:
            ck = c0
        else:
            ck = np.sum(demeaned_series[:-k] * demeaned_series[k:]) / N
        acf_values.append(ck / c0)
    
    return acf_values

def calculate_pacf(y, lags):
    """Calculate Partial Autocorrelation Function"""
    rho = calculate_acf(y, lags)
    pacf_vals = [1.0]  # PACF(0)=1
    
    for k in range(1, lags + 1):
        P_k = np.array([[rho[abs(i - j)] for j in range(k)] for i in range(k)])
        rho_k = np.array(rho[1 : k + 1])
        P_k += np.eye(k) * 1e-8  # Add small epsilon for stability
        phi_k = np.linalg.solve(P_k, rho_k)
        pacf_vals.append(phi_k[-1])
    
    return np.array(pacf_vals)

def fit_ar_yule_walker(y, p):
    """Fit AR model using Yule-Walker equations"""
    y = y - np.mean(y)
    n = len(y)
    if p == 0:
        return np.array([]), np.var(y)
    
    r = [np.sum(y[:n - k] * y[k:]) / n for k in range(p + 1)]
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    rhs = np.array(r[1:])
    
    try:
        phi = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        phi = np.zeros(p)
    
    sigma2 = max(r[0] - np.dot(phi, rhs), 1e-8)
    return phi, max(sigma2, 1e-8)

def arma_negloglike(params, y, p, q):
    """ARMA negative log-likelihood function"""
    phi = params[:p] if p > 0 else np.array([])
    theta = params[p:p + q] if q > 0 else np.array([])
    sigma2 = np.exp(params[-1])
    n = len(y)
    eps = np.zeros(n)
    
    for t in range(n):
        ar = sum(phi[i] * y[t - 1 - i] for i in range(p) if t - 1 - i >= 0)
        ma = sum(theta[j] * eps[t - 1 - j] for j in range(q) if t - 1 - j >= 0)
        eps[t] = y[t] - ar - ma
    
    nll = 0.5 * n * np.log(2 * np.pi * sigma2) + 0.5 * np.sum(eps**2) / sigma2
    return nll

def fit_arima_manual(y, p, q, maxiter=300):
    """Fit ARIMA model manually"""
    y = np.array(y) - np.mean(y)
    phi_init, s2_init = fit_ar_yule_walker(y, p)
    theta_init = np.zeros(q)
    init = np.concatenate([phi_init, theta_init, [np.log(s2_init)]])
    
    res = optimize.minimize(
        arma_negloglike, init, args=(y, p, q),
        method="L-BFGS-B", options={"maxiter": maxiter, "disp": False}
    )
    
    est = res.x
    phi_hat = est[:p]
    theta_hat = est[p:p + q]
    sigma2_hat = np.exp(est[-1])
    nll = res.fun
    k = len(est)
    aic = 2 * k + 2 * nll
    bic = k * np.log(len(y)) + 2 * nll
    
    return {
        "phi": phi_hat, 
        "theta": theta_hat, 
        "sigma2": sigma2_hat, 
        "aic": aic, 
        "bic": bic, 
        "nll": nll
    }

def ols_manual(x, y):
    """Manual OLS regression"""
    x = np.asarray(x)
    y = np.asarray(y)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm)**2)
    if denom == 0:
        return 0.0, 0.0, np.nan
    
    beta = np.sum((x - xm)*(y - ym)) / denom
    alpha = ym - beta * xm
    resid = y - (alpha + beta*x)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - ym)**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
    
    return alpha, beta, r2

def portfolio_return(weights, mean_returns):
    """Calculate portfolio return"""
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights, cov_matrix):
    """Calculate portfolio volatility"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# ===============================================================
# PAGE 1: HOME & DATA LOADING
# ===============================================================

if page == "🏠 Home & Data Loading":
    st.markdown('<h1 class="main-header">Mathematical Modeling in Finance</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This comprehensive application implements advanced mathematical models for financial analysis:
    
    - **📈 Time Series Analysis**: ACF/PACF analysis and ARIMA modeling
    - **💼 Portfolio Optimization**: Modern Portfolio Theory (Markowitz)
    - **📊 CAPM Analysis**: Capital Asset Pricing Model with Bitcoin as market proxy
    - **🎯 Simple Index Model**: Single-factor portfolio optimization
    - **⚖️ Model Comparison**: Comprehensive evaluation framework
    """)
    
    st.markdown('<h2 class="section-header">📊 Data Configuration</h2>', unsafe_allow_html=True)
    
    # Data loading section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Cryptocurrency Selection")
        default_tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
        
        # Allow users to modify tickers
        tickers_input = st.text_area(
            "Enter cryptocurrency tickers (one per line):",
            value="\n".join(default_tickers),
            height=150
        )
        
        tickers = [ticker.strip() for ticker in tickers_input.split('\n') if ticker.strip()]
        
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2017-01-01"),
            min_value=pd.to_datetime("2010-01-01"),
            max_value=pd.to_datetime("2024-01-01")
        )
    
    with col2:
        st.subheader("Data Statistics")
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded Successfully")
            st.info(f"📊 Assets: {len(st.session_state.company_returns)}")
            if st.session_state.returns_data is not None:
                st.info(f"📅 Date Range: {st.session_state.returns_data.index.min().strftime('%Y-%m-%d')} to {st.session_state.returns_data.index.max().strftime('%Y-%m-%d')}")
                st.info(f"📈 Observations: {len(st.session_state.returns_data)}")
        else:
            st.warning("⚠️ No data loaded yet")
    
    # Load data button
    if st.button("🔄 Load/Refresh Data", type="primary"):
        with st.spinner("Downloading cryptocurrency data..."):
            combined_data, returns_matrix, company_returns = download_crypto_data(tickers, start_date.strftime('%Y-%m-%d'))
            
            if combined_data is not None:
                # Debug information
                st.info(f"📊 Data loaded: {len(company_returns)} assets")
                st.info(f"📅 Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
                st.info(f"🏷️ Assets: {list(returns_matrix.columns)}")
                
                st.session_state.combined_data = combined_data
                st.session_state.returns_data = returns_matrix
                st.session_state.company_returns = company_returns
                st.session_state.data_loaded = True
                st.success(f"✅ Successfully loaded data for {len(company_returns)} assets!")
                st.rerun()
            else:
                st.error("❌ Failed to load data. Please check your internet connection and ticker symbols.")
    
    # Display data preview if loaded
    if st.session_state.data_loaded and st.session_state.returns_data is not None:
        st.markdown('<h2 class="section-header">📈 Data Preview</h2>', unsafe_allow_html=True)
        
        # Interactive plot of returns
        fig = go.Figure()
        
        try:
            for ticker in st.session_state.returns_data.columns:
                # Ensure ticker name is a string and clean
                if isinstance(ticker, (tuple, list)):
                    # If it's a tuple/list, take the last non-empty element
                    ticker_name = str([x for x in ticker if x][-1]) if ticker else "Unknown"
                else:
                    ticker_name = str(ticker).strip() if ticker else "Unknown"
                
                if ticker_name and ticker_name != "":
                    fig.add_trace(go.Scatter(
                        x=st.session_state.returns_data.index,
                        y=st.session_state.returns_data[ticker],
                        mode='lines',
                        name=ticker_name,
                        line=dict(width=1.5)
                    ))
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
            st.write("Debug info:")
            st.write(f"Columns: {list(st.session_state.returns_data.columns)}")
            st.write(f"Column types: {[type(col) for col in st.session_state.returns_data.columns]}")
        
        fig.update_layout(
            title="Daily Log Returns for All Cryptocurrencies",
            xaxis_title="Date",
            yaxis_title="Log Return",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        summary_stats = st.session_state.returns_data.describe()
        st.dataframe(summary_stats.round(6))
        
        # Correlation matrix
        st.subheader("🔗 Correlation Matrix")
        corr_matrix = st.session_state.returns_data.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Returns Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ===============================================================
# PAGE 2: TIME SERIES ANALYSIS
# ===============================================================

elif page == "📈 Time Series Analysis":
    st.markdown('<h1 class="main-header">Time Series Analysis</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page.")
        st.stop()
    
    st.markdown('<h2 class="section-header">ACF & PACF Analysis</h2>', unsafe_allow_html=True)
    
    # Asset selection
    selected_asset = st.selectbox(
        "Select Asset for Analysis:",
        options=list(st.session_state.company_returns.keys())
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_lags = st.slider("Maximum Lags", min_value=10, max_value=100, value=40)
    
    with col2:
        confidence_level = st.slider("Confidence Level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    
    if selected_asset and st.button("🔍 Analyze ACF/PACF"):
        y = st.session_state.company_returns[selected_asset]
        y_demeaned = y - np.mean(y)
        
        # Calculate ACF and PACF
        acf_vals = calculate_acf(y_demeaned, max_lags)
        pacf_vals = calculate_pacf(y_demeaned, max_lags)
        
        # Confidence bounds
        n = len(y)
        conf_bound = stats.norm.ppf((1 + confidence_level) / 2) / np.sqrt(n)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)')
        )
        
        # ACF plot
        lags = list(range(len(acf_vals)))
        fig.add_trace(
            go.Scatter(x=lags, y=acf_vals, mode='markers+lines', name='ACF', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=0, line_color="black", row=1, col=1)
        
        # PACF plot
        lags_pacf = list(range(len(pacf_vals)))
        fig.add_trace(
            go.Scatter(x=lags_pacf, y=pacf_vals, mode='markers+lines', name='PACF', line=dict(color='orange')),
            row=1, col=2
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=0, line_color="black", row=1, col=2)
        
        fig.update_layout(height=500, title_text=f"ACF & PACF Analysis for {selected_asset}")
        fig.update_xaxes(title_text="Lag", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)
        fig.update_yaxes(title_text="ACF", row=1, col=1)
        fig.update_yaxes(title_text="PACF", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Suggest ARIMA orders
        sig_acf = [lag for lag, val in enumerate(acf_vals) if abs(val) > conf_bound and lag > 0]
        sig_pacf = [lag for lag, val in enumerate(pacf_vals) if abs(val) > conf_bound and lag > 0]
        
        p_suggested = max(sig_pacf) if len(sig_pacf) > 0 else 0
        q_suggested = max(sig_acf) if len(sig_acf) > 0 else 0
        
        st.markdown('<h3 class="section-header">📋 ARIMA Order Suggestions</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Suggested p (AR)", p_suggested)
        with col2:
            st.metric("Suggested d (I)", 0, help="Assuming stationary returns")
        with col3:
            st.metric("Suggested q (MA)", q_suggested)
        
        # Significant lags table
        st.subheader("📊 Significant Lags")
        
        # Ensure both lists have the same length for DataFrame creation
        max_lags_to_show = 10
        acf_lags_display = (sig_acf[:max_lags_to_show] if sig_acf else []) + [''] * max_lags_to_show
        pacf_lags_display = (sig_pacf[:max_lags_to_show] if sig_pacf else []) + [''] * max_lags_to_show
        
        # Truncate to exactly max_lags_to_show elements
        acf_lags_display = acf_lags_display[:max_lags_to_show]
        pacf_lags_display = pacf_lags_display[:max_lags_to_show]
        
        # Replace empty strings with 'None' for better display
        acf_lags_display = [str(lag) if lag != '' else 'None' for lag in acf_lags_display]
        pacf_lags_display = [str(lag) if lag != '' else 'None' for lag in pacf_lags_display]
        
        sig_lags_df = pd.DataFrame({
            'ACF Significant Lags': acf_lags_display,
            'PACF Significant Lags': pacf_lags_display
        })
        st.dataframe(sig_lags_df)
    
    # Batch analysis for all assets
    st.markdown('<h2 class="section-header">🔄 Batch Analysis</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Analyze All Assets"):
        results = []
        progress_bar = st.progress(0)
        
        for i, (ticker, returns) in enumerate(st.session_state.company_returns.items()):
            y = returns - np.mean(returns)
            n = len(y)
            conf_bound = 1.96 / np.sqrt(n)
            
            acf_vals = calculate_acf(y, min(40, n//4))
            pacf_vals = calculate_pacf(y, min(40, n//4))
            
            sig_acf = [lag for lag, val in enumerate(acf_vals) if abs(val) > conf_bound and lag > 0]
            sig_pacf = [lag for lag, val in enumerate(pacf_vals) if abs(val) > conf_bound and lag > 0]
            
            p = max(sig_pacf) if len(sig_pacf) > 0 else 0
            q = max(sig_acf) if len(sig_acf) > 0 else 0
            
            results.append({
                "Asset": ticker,
                "Observations": n,
                "Suggested p": p,
                "Suggested q": q,
                "Confidence Bound": f"±{conf_bound:.4f}",
                "Significant ACF Lags": len(sig_acf),
                "Significant PACF Lags": len(sig_pacf)
            })
            
            progress_bar.progress((i + 1) / len(st.session_state.company_returns))
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Summary visualization
        fig_summary = px.scatter(
            results_df, 
            x="Suggested p", 
            y="Suggested q",
            size="Observations",
            hover_data=["Asset"],
            title="Suggested ARIMA Orders for All Assets"
        )
        st.plotly_chart(fig_summary, use_container_width=True)

# ===============================================================
# PAGE 3: ARIMA MODELING
# ===============================================================

elif page == "🔍 ARIMA Modeling":
    st.markdown('<h1 class="main-header">ARIMA Modeling & Forecasting</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page.")
        st.stop()
    
    st.markdown('<h2 class="section-header">🎯 Model Configuration</h2>', unsafe_allow_html=True)
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        selected_asset = st.selectbox(
            "Select Asset for ARIMA Modeling:",
            options=list(st.session_state.company_returns.keys())
        )
        
        p_order = st.slider("AR Order (p)", min_value=0, max_value=10, value=2)
        q_order = st.slider("MA Order (q)", min_value=0, max_value=10, value=2)
    
    with col2:
        max_iterations = st.slider("Max Iterations", min_value=100, max_value=1000, value=300)
        forecast_steps = st.slider("Forecast Steps", min_value=1, max_value=30, value=5)
    
    if st.button("🚀 Fit ARIMA Model"):
        if selected_asset:
            y = st.session_state.company_returns[selected_asset]
            
            with st.spinner(f"Fitting ARIMA({p_order},0,{q_order}) model..."):
                try:
                    # Fit ARIMA model
                    fit_result = fit_arima_manual(y, p_order, q_order, max_iterations)
                    
                    # Display results
                    st.success(f"✅ ARIMA({p_order},0,{q_order}) model fitted successfully!")
                    
                    # Model statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("AIC", f"{fit_result['aic']:.2f}")
                    with col2:
                        st.metric("BIC", f"{fit_result['bic']:.2f}")
                    with col3:
                        st.metric("Log-Likelihood", f"{-fit_result['nll']:.2f}")
                    with col4:
                        st.metric("Residual Variance", f"{fit_result['sigma2']:.6f}")
                    
                    # Parameter estimates
                    st.header("📊 Parameter Estimates")
                    
                    param_data = []
                    
                    # AR parameters
                    for i, phi in enumerate(fit_result['phi']):
                        param_data.append({
                            'Parameter': f'φ_{i+1} (AR)',
                            'Estimate': phi,
                            'Type': 'Autoregressive'
                        })
                    
                    # MA parameters
                    for i, theta in enumerate(fit_result['theta']):
                        param_data.append({
                            'Parameter': f'θ_{i+1} (MA)',
                            'Estimate': theta,
                            'Type': 'Moving Average'
                        })
                    
                    if param_data:
                        param_df = pd.DataFrame(param_data)
                        st.dataframe(param_df, use_container_width=True)
                    
                    # Calculate residuals
                    y_centered = y - np.mean(y)
                    n = len(y_centered)
                    eps = np.zeros(n)
                    
                    for t in range(n):
                        ar_term = sum(fit_result['phi'][i] * y_centered[t - 1 - i] 
                                    for i in range(len(fit_result['phi'])) if t - 1 - i >= 0)
                        ma_term = sum(fit_result['theta'][j] * eps[t - 1 - j] 
                                    for j in range(len(fit_result['theta'])) if t - 1 - j >= 0)
                        eps[t] = y_centered[t] - ar_term - ma_term
                    
                    # Residual analysis
                    st.subheader("📈 Residual Analysis")
                    
                    fig_residuals = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Residuals Time Series', 'Residuals Histogram', 
                                      'Q-Q Plot', 'ACF of Residuals')
                    )
                    
                    # Residuals time series
                    fig_residuals.add_trace(
                        go.Scatter(y=eps, mode='lines', name='Residuals', line=dict(color='blue')),
                        row=1, col=1
                    )
                    
                    # Residuals histogram
                    fig_residuals.add_trace(
                        go.Histogram(x=eps, name='Residuals Hist', nbinsx=30),
                        row=1, col=2
                    )
                    
                    # Q-Q plot (simplified)
                    sorted_residuals = np.sort(eps)
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
                    fig_residuals.add_trace(
                        go.Scatter(x=theoretical_quantiles, y=sorted_residuals, 
                                 mode='markers', name='Q-Q Plot'),
                        row=2, col=1
                    )
                    
                    # ACF of residuals
                    residual_acf = calculate_acf(eps, min(20, len(eps)//4))
                    fig_residuals.add_trace(
                        go.Scatter(y=residual_acf, mode='markers+lines', name='Residual ACF'),
                        row=2, col=2
                    )
                    
                    fig_residuals.update_layout(height=600, title_text="Residual Diagnostics")
                    st.plotly_chart(fig_residuals, use_container_width=True)
                    
                    # Rolling forecast validation
                    st.subheader("🎯 Rolling Forecast Validation")
                    
                    window_size = min(60, len(y) // 3)
                    forecasts = []
                    actuals = []
                    
                    for i in range(window_size, len(y) - 1):
                        train_data = y[i - window_size:i]
                        actual = y[i]
                        
                        try:
                            # Simple forecast (using last value + AR component)
                            train_centered = train_data - np.mean(train_data)
                            forecast = np.mean(train_data)
                            
                            # Add AR component if available
                            if len(fit_result['phi']) > 0:
                                ar_component = sum(fit_result['phi'][j] * train_centered[-1-j] 
                                                 for j in range(min(len(fit_result['phi']), len(train_centered))))
                                forecast += ar_component
                            
                            forecasts.append(forecast)
                            actuals.append(actual)
                            
                        except:
                            continue
                    
                    if forecasts:
                        forecasts = np.array(forecasts)
                        actuals = np.array(actuals)
                        
                        # Calculate metrics
                        rmse = np.sqrt(np.mean((forecasts - actuals) ** 2))
                        mae = np.mean(np.abs(forecasts - actuals))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("RMSE", f"{rmse:.6f}")
                        with col2:
                            st.metric("MAE", f"{mae:.6f}")
                        
                        # Plot forecast vs actual
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(y=actuals, mode='lines', name='Actual', line=dict(color='blue')))
                        fig_forecast.add_trace(go.Scatter(y=forecasts, mode='lines', name='Forecast', line=dict(color='red')))
                        fig_forecast.update_layout(
                            title="Rolling One-Step-Ahead Forecasts",
                            yaxis_title="Returns",
                            xaxis_title="Time Index"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Store results in session state for comparison
                    if 'arima_results' not in st.session_state:
                        st.session_state.arima_results = {}
                    
                    st.session_state.arima_results[selected_asset] = {
                        'model': fit_result,
                        'p': p_order,
                        'q': q_order,
                        'residuals': eps,
                        'rmse': rmse if 'rmse' in locals() else None,
                        'mae': mae if 'mae' in locals() else None
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error fitting ARIMA model: {str(e)}")
    
    # Batch ARIMA fitting
    st.markdown('<h2 class="section-header">🔄 Batch ARIMA Analysis</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Fit ARIMA for All Assets"):
        batch_results = []
        progress_bar = st.progress(0)
        
        for i, (ticker, returns) in enumerate(st.session_state.company_returns.items()):
            try:
                # Use default orders for batch processing
                fit_result = fit_arima_manual(returns, 2, 2, 200)
                
                batch_results.append({
                    'Asset': ticker,
                    'AIC': fit_result['aic'],
                    'BIC': fit_result['bic'],
                    'Log-Likelihood': -fit_result['nll'],
                    'Residual Variance': fit_result['sigma2'],
                    'AR Parameters': len(fit_result['phi']),
                    'MA Parameters': len(fit_result['theta'])
                })
                
            except Exception as e:
                batch_results.append({
                    'Asset': ticker,
                    'AIC': np.nan,
                    'BIC': np.nan,
                    'Log-Likelihood': np.nan,
                    'Residual Variance': np.nan,
                    'AR Parameters': 0,
                    'MA Parameters': 0
                })
            
            progress_bar.progress((i + 1) / len(st.session_state.company_returns))
        
        batch_df = pd.DataFrame(batch_results)
        st.dataframe(batch_df, use_container_width=True)
        
        # Visualization of model performance
        fig_batch = px.scatter(
            batch_df.dropna(), 
            x='AIC', 
            y='BIC',
            hover_data=['Asset'],
            title='ARIMA Model Performance (AIC vs BIC)'
        )
        st.plotly_chart(fig_batch, use_container_width=True)

# ===============================================================
# PAGE 4: PORTFOLIO OPTIMIZATION
# ===============================================================

elif page == "💼 Portfolio Optimization":
    st.markdown('<h1 class="main-header">Portfolio Optimization (Markowitz)</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page.")
        st.stop()
    
    st.markdown('<h2 class="section-header">⚙️ Optimization Parameters</h2>', unsafe_allow_html=True)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        risk_free_rate = st.slider("Risk-Free Rate (Annual %)", min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100
        trading_days = st.slider("Trading Days per Year", min_value=200, max_value=365, value=252)
    
    with col2:
        frontier_points = st.slider("Efficient Frontier Points", min_value=50, max_value=500, value=100)
        allow_short_selling = st.checkbox("Allow Short Selling", value=False)
    
    if st.button("🚀 Optimize Portfolio"):
        # Prepare data
        returns_df = st.session_state.returns_data.dropna()
        
        if returns_df.empty:
            st.error("❌ No valid returns data available.")
            st.stop()
        
        # Calculate annualized statistics
        mean_returns = returns_df.mean() * trading_days
        cov_matrix = returns_df.cov() * trading_days
        
        st.success(f"✅ Optimizing portfolio with {len(returns_df.columns)} assets")
        
        # Display input statistics
        st.subheader("📊 Input Statistics (Annualized)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Expected Returns:**")
            returns_display = pd.DataFrame({
                'Asset': mean_returns.index,
                'Expected Return (%)': (mean_returns * 100).round(2)
            })
            st.dataframe(returns_display, use_container_width=True)
        
        with col2:
            st.write("**Volatilities:**")
            volatilities = np.sqrt(np.diag(cov_matrix))
            vol_display = pd.DataFrame({
                'Asset': mean_returns.index,
                'Volatility (%)': (volatilities * 100).round(2)
            })
            st.dataframe(vol_display, use_container_width=True)
        
        # Portfolio optimization functions
        def objective_volatility(weights, mean_returns, cov_matrix):
            return portfolio_volatility(weights, cov_matrix)
        
        def objective_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            p_return = portfolio_return(weights, mean_returns)
            p_volatility = portfolio_volatility(weights, cov_matrix)
            if p_volatility == 0:
                return 1e10
            return -(p_return - risk_free_rate) / p_volatility
        
        # Constraints and bounds
        num_assets = len(mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        if allow_short_selling:
            bounds = tuple((-1, 1) for _ in range(num_assets))
        else:
            bounds = tuple((0, 1) for _ in range(num_assets))
        
        initial_weights = np.array(num_assets * [1. / num_assets])
        
        # Find minimum volatility portfolio
        with st.spinner("Finding minimum volatility portfolio..."):
            min_vol_result = optimize.minimize(
                objective_volatility, initial_weights, 
                args=(mean_returns, cov_matrix),
                method='SLSQP', bounds=bounds, constraints=constraints
            )
        
        min_vol_weights = min_vol_result.x
        min_vol_return = portfolio_return(min_vol_weights, mean_returns)
        min_vol_std = portfolio_volatility(min_vol_weights, cov_matrix)
        
        # Find maximum Sharpe ratio portfolio
        with st.spinner("Finding maximum Sharpe ratio portfolio..."):
            max_sharpe_result = optimize.minimize(
                objective_sharpe_ratio, initial_weights,
                args=(mean_returns, cov_matrix, risk_free_rate),
                method='SLSQP', bounds=bounds, constraints=constraints
            )
        
        max_sharpe_weights = max_sharpe_result.x
        max_sharpe_return = portfolio_return(max_sharpe_weights, mean_returns)
        max_sharpe_std = portfolio_volatility(max_sharpe_weights, cov_matrix)
        max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_std
        
        # Generate efficient frontier
        with st.spinner("Generating efficient frontier..."):
            min_ret = mean_returns.min() * 0.8
            max_ret = mean_returns.max() * 1.2
            target_returns = np.linspace(min_ret, max_ret, frontier_points)
            
            efficient_portfolios = []
            
            for target in target_returns:
                constraints_target = (
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                    {'type': 'eq', 'fun': lambda weights: portfolio_return(weights, mean_returns) - target}
                )
                
                try:
                    result = optimize.minimize(
                        objective_volatility, initial_weights,
                        args=(mean_returns, cov_matrix),
                        method='SLSQP', bounds=bounds, constraints=constraints_target,
                        options={'maxiter': 1000}
                    )
                    
                    if result.success:
                        efficient_portfolios.append({
                            "Return": target,
                            "Volatility": portfolio_volatility(result.x, cov_matrix),
                            "Weights": result.x
                        })
                except:
                    continue
        
        if not efficient_portfolios:
            st.error("❌ Failed to generate efficient frontier.")
            st.stop()
        
        efficient_df = pd.DataFrame(efficient_portfolios).sort_values(by="Volatility")
        
        # Display key portfolios
        st.markdown('<h2 class="section-header">🎯 Key Portfolios</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛡️ Minimum Volatility Portfolio")
            st.metric("Expected Return", f"{min_vol_return*100:.2f}%")
            st.metric("Volatility", f"{min_vol_std*100:.2f}%")
            
            min_vol_weights_df = pd.DataFrame({
                'Asset': mean_returns.index,
                'Weight (%)': (min_vol_weights * 100).round(2)
            }).sort_values('Weight (%)', ascending=False)
            st.dataframe(min_vol_weights_df, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Maximum Sharpe Ratio Portfolio")
            st.metric("Expected Return", f"{max_sharpe_return*100:.2f}%")
            st.metric("Volatility", f"{max_sharpe_std*100:.2f}%")
            st.metric("Sharpe Ratio", f"{max_sharpe_ratio:.3f}")
            
            max_sharpe_weights_df = pd.DataFrame({
                'Asset': mean_returns.index,
                'Weight (%)': (max_sharpe_weights * 100).round(2)
            }).sort_values('Weight (%)', ascending=False)
            st.dataframe(max_sharpe_weights_df, use_container_width=True)
        
        # Efficient frontier plot
        st.markdown('<h2 class="section-header">📈 Efficient Frontier</h2>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=efficient_df['Volatility'] * 100,
            y=efficient_df['Return'] * 100,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3)
        ))
        
        # Individual assets
        # Ensure text labels are strings
        asset_labels = [str(asset) for asset in mean_returns.index]
        fig.add_trace(go.Scatter(
            x=volatilities * 100,
            y=mean_returns * 100,
            mode='markers+text',
            name='Individual Assets',
            text=asset_labels,
            textposition="top center",
            marker=dict(size=10, color='red')
        ))
        
        # Key portfolios
        fig.add_trace(go.Scatter(
            x=[min_vol_std * 100],
            y=[min_vol_return * 100],
            mode='markers',
            name='Min Volatility',
            marker=dict(size=15, color='green', symbol='star')
        ))
        
        fig.add_trace(go.Scatter(
            x=[max_sharpe_std * 100],
            y=[max_sharpe_return * 100],
            mode='markers',
            name='Max Sharpe',
            marker=dict(size=15, color='gold', symbol='star')
        ))
        
        # Capital Market Line
        cml_x = np.linspace(0, efficient_df['Volatility'].max() * 1.1 * 100, 100)
        cml_y = risk_free_rate * 100 + max_sharpe_ratio * cml_x
        fig.add_trace(go.Scatter(
            x=cml_x,
            y=cml_y,
            mode='lines',
            name='Capital Market Line',
            line=dict(color='green', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title='Efficient Frontier and Capital Market Line',
            xaxis_title='Volatility (%)',
            yaxis_title='Expected Return (%)',
            hovermode='closest',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio weights evolution
        st.subheader("🔄 Portfolio Weights Evolution")
        
        weights_df = pd.DataFrame([p['Weights'] for p in efficient_portfolios])
        weights_df.columns = mean_returns.index
        weights_df['Volatility'] = [p['Volatility'] for p in efficient_portfolios]
        
        fig_weights = go.Figure()
        
        for asset in mean_returns.index:
            # Ensure asset name is a string
            asset_name = str(asset) if not isinstance(asset, str) else asset
            fig_weights.add_trace(go.Scatter(
                x=weights_df['Volatility'] * 100,
                y=weights_df[asset] * 100,
                mode='lines',
                name=asset_name,
                stackgroup='one' if not allow_short_selling else None
            ))
        
        fig_weights.update_layout(
            title='Portfolio Weights vs Risk Level',
            xaxis_title='Portfolio Volatility (%)',
            yaxis_title='Asset Weight (%)',
            height=500
        )
        
        st.plotly_chart(fig_weights, use_container_width=True)
        
        # Store results for comparison
        if 'portfolio_results' not in st.session_state:
            st.session_state.portfolio_results = {}
        
        st.session_state.portfolio_results['markowitz'] = {
            'efficient_frontier': efficient_df,
            'min_vol_portfolio': {
                'weights': min_vol_weights,
                'return': min_vol_return,
                'volatility': min_vol_std
            },
            'max_sharpe_portfolio': {
                'weights': max_sharpe_weights,
                'return': max_sharpe_return,
                'volatility': max_sharpe_std,
                'sharpe_ratio': max_sharpe_ratio
            },
            'mean_returns': mean_returns,
            'cov_matrix': cov_matrix
        }

# ===============================================================
# PAGE 5: CAPM ANALYSIS
# ===============================================================

elif page == "📊 CAPM Analysis":
    st.markdown('<h1 class="main-header">Capital Asset Pricing Model (CAPM)</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page.")
        st.stop()
    
    st.markdown('<h2 class="section-header">🎯 CAPM Configuration</h2>', unsafe_allow_html=True)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        # Market proxy selection
        market_options = list(st.session_state.returns_data.columns)
        btc_default = next((col for col in market_options if 'BTC' in str(col)), market_options[0])
        
        market_proxy = st.selectbox(
            "Select Market Proxy:",
            options=market_options,
            index=market_options.index(btc_default) if btc_default in market_options else 0
        )
        
        risk_free_rate_annual = st.slider("Risk-Free Rate (Annual %)", min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100
    
    with col2:
        trading_days = st.slider("Trading Days per Year", min_value=200, max_value=365, value=252)
        rolling_window = st.slider("Rolling Beta Window (Days)", min_value=30, max_value=365, value=180)
    
    # Calculate daily risk-free rate
    rf_daily = (1 + risk_free_rate_annual)**(1/trading_days) - 1
    
    if st.button("🚀 Run CAPM Analysis"):
        returns_data = st.session_state.returns_data.dropna()
        
        # Market returns
        market_returns = returns_data[market_proxy].values
        market_excess = market_returns - rf_daily
        
        # Assets (excluding market proxy)
        assets = [col for col in returns_data.columns if col != market_proxy]
        
        if not assets:
            st.error("❌ No assets available for CAPM analysis (all assets are market proxy).")
            st.stop()
        
        st.success(f"✅ Running CAPM analysis with {market_proxy} as market proxy")
        
        # CAPM estimation for each asset
        capm_results = []
        
        for asset in assets:
            asset_returns = returns_data[asset].values
            asset_excess = asset_returns - rf_daily
            
            # OLS regression
            alpha, beta, r2 = ols_manual(market_excess, asset_excess)
            
            # Calculate statistics
            n_obs = len(asset_excess)
            residuals = asset_excess - (alpha + beta * market_excess)
            ss_res = np.sum(residuals**2)
            s2 = ss_res / (n_obs - 2)
            
            # Standard errors
            se_beta = np.sqrt(s2 / np.sum((market_excess - market_excess.mean())**2))
            se_alpha = np.sqrt(s2 * (1/n_obs + market_excess.mean()**2 / np.sum((market_excess - market_excess.mean())**2)))
            
            # t-statistics and p-values
            t_beta = beta / se_beta if se_beta > 0 else 0
            t_alpha = alpha / se_alpha if se_alpha > 0 else 0
            p_beta = 2 * stats.t.sf(np.abs(t_beta), df=n_obs-2)
            p_alpha = 2 * stats.t.sf(np.abs(t_alpha), df=n_obs-2)
            
            # Annualized metrics
            ann_return = (1 + asset_returns.mean())**trading_days - 1
            ann_vol = asset_returns.std() * np.sqrt(trading_days)
            market_ann_return = (1 + market_returns.mean())**trading_days - 1
            capm_predicted = risk_free_rate_annual + beta * (market_ann_return - risk_free_rate_annual)
            
            # Error metrics
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            
            capm_results.append({
                'Asset': asset,
                'Alpha (daily)': alpha,
                'Beta': beta,
                'R²': r2,
                'SE_Alpha': se_alpha,
                'SE_Beta': se_beta,
                't_Alpha': t_alpha,
                't_Beta': t_beta,
                'p_Alpha': p_alpha,
                'p_Beta': p_beta,
                'Ann_Return (%)': ann_return * 100,
                'Ann_Vol (%)': ann_vol * 100,
                'CAPM_Predicted (%)': capm_predicted * 100,
                'RMSE': rmse,
                'MAE': mae,
                'Observations': n_obs
            })
        
        capm_df = pd.DataFrame(capm_results)
        
        # Display results
        st.markdown('<h2 class="section-header">📊 CAPM Results</h2>', unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Beta", f"{capm_df['Beta'].mean():.3f}")
        with col2:
            st.metric("Average R²", f"{capm_df['R²'].mean():.3f}")
        with col3:
            st.metric("Significant Betas", f"{(capm_df['p_Beta'] < 0.05).sum()}/{len(capm_df)}")
        with col4:
            st.metric("Average RMSE", f"{capm_df['RMSE'].mean():.6f}")
        
        # Detailed results table
        st.subheader("📋 Detailed CAPM Results")
        
        # Format the dataframe for display
        display_df = capm_df.copy()
        numeric_cols = ['Alpha (daily)', 'Beta', 'R²', 'SE_Alpha', 'SE_Beta', 't_Alpha', 't_Beta', 'RMSE', 'MAE']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(6)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Security Characteristic Lines (SCL)
        st.markdown('<h2 class="section-header">📈 Security Characteristic Lines</h2>', unsafe_allow_html=True)
        
        # Create subplots for SCL
        n_assets = len(assets)
        cols = 2
        rows = (n_assets + cols - 1) // cols
        
        fig_scl = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"SCL: {asset}" for asset in assets],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, asset in enumerate(assets):
            row = i // cols + 1
            col = i % cols + 1
            
            asset_excess = returns_data[asset].values - rf_daily
            alpha = capm_df[capm_df['Asset'] == asset]['Alpha (daily)'].iloc[0]
            beta = capm_df[capm_df['Asset'] == asset]['Beta'].iloc[0]
            r2 = capm_df[capm_df['Asset'] == asset]['R²'].iloc[0]
            
            # Regression line
            x_line = np.linspace(market_excess.min(), market_excess.max(), 100)
            y_line = alpha + beta * x_line
            
            # Scatter plot
            fig_scl.add_trace(
                go.Scatter(
                    x=market_excess,
                    y=asset_excess,
                    mode='markers',
                    name=f'{asset} Returns',
                    marker=dict(size=3, opacity=0.6),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Regression line
            fig_scl.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f'β={beta:.3f}, R²={r2:.3f}',
                    line=dict(color='red', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig_scl.update_layout(
            height=300 * rows,
            title_text="Security Characteristic Lines (SCL)",
            showlegend=False
        )
        
        # Update axes labels
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig_scl.update_xaxes(title_text="Market Excess Return", row=i, col=j)
                fig_scl.update_yaxes(title_text="Asset Excess Return", row=i, col=j)
        
        st.plotly_chart(fig_scl, use_container_width=True)
        
        # Beta analysis
        st.subheader("🎯 Beta Analysis")
        
        fig_beta = px.bar(
            capm_df, 
            x='Asset', 
            y='Beta',
            title='Beta Coefficients by Asset',
            color='Beta',
            color_continuous_scale='RdYlBu_r'
        )
        fig_beta.add_hline(y=1, line_dash="dash", line_color="black", annotation_text="Market Beta = 1")
        st.plotly_chart(fig_beta, use_container_width=True)
        
        # Actual vs Predicted Returns
        st.subheader("🎯 Actual vs CAPM Predicted Returns")
        
        fig_pred = px.scatter(
            capm_df,
            x='Ann_Return (%)',
            y='CAPM_Predicted (%)',
            hover_data=['Asset', 'Beta', 'R²'],
            title='Actual vs CAPM Predicted Annual Returns'
        )
        
        # Add diagonal line
        min_ret = min(capm_df['Ann_Return (%)'].min(), capm_df['CAPM_Predicted (%)'].min())
        max_ret = max(capm_df['Ann_Return (%)'].max(), capm_df['CAPM_Predicted (%)'].max())
        fig_pred.add_trace(
            go.Scatter(
                x=[min_ret, max_ret],
                y=[min_ret, max_ret],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Rolling Beta Analysis
        st.markdown('<h2 class="section-header">📊 Rolling Beta Analysis</h2>', unsafe_allow_html=True)
        
        if st.button("📈 Calculate Rolling Betas"):
            rolling_betas = {}
            
            progress_bar = st.progress(0)
            
            for i, asset in enumerate(assets):
                asset_excess = returns_data[asset].values - rf_daily
                rolling_beta_series = []
                dates = []
                
                for j in range(rolling_window, len(returns_data)):
                    window_market = market_excess[j-rolling_window:j]
                    window_asset = asset_excess[j-rolling_window:j]
                    
                    _, beta, _ = ols_manual(window_market, window_asset)
                    rolling_beta_series.append(beta)
                    dates.append(returns_data.index[j])
                
                rolling_betas[asset] = {
                    'dates': dates,
                    'betas': rolling_beta_series
                }
                
                progress_bar.progress((i + 1) / len(assets))
            
            # Plot rolling betas
            fig_rolling = go.Figure()
            
            for asset in assets:
                fig_rolling.add_trace(
                    go.Scatter(
                        x=rolling_betas[asset]['dates'],
                        y=rolling_betas[asset]['betas'],
                        mode='lines',
                        name=asset,
                        line=dict(width=2)
                    )
                )
            
            fig_rolling.add_hline(y=1, line_dash="dash", line_color="black", annotation_text="Market Beta = 1")
            fig_rolling.update_layout(
                title=f'Rolling {rolling_window}-Day Betas',
                xaxis_title='Date',
                yaxis_title='Beta',
                height=500
            )
            
            st.plotly_chart(fig_rolling, use_container_width=True)
        
        # Store results for comparison
        if 'capm_results' not in st.session_state:
            st.session_state.capm_results = {}
        
        st.session_state.capm_results = {
            'results_df': capm_df,
            'market_proxy': market_proxy,
            'risk_free_rate': risk_free_rate_annual,
            'market_returns': market_returns,
            'market_excess': market_excess
        }

# ===============================================================
# PAGE 6: SIMPLE INDEX MODEL
# ===============================================================

elif page == "🎯 Simple Index Model":
    st.markdown('<h1 class="main-header">Simple Index Model (SIM)</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page.")
        st.stop()
    
    st.markdown('<h2 class="section-header">⚙️ SIM Configuration</h2>', unsafe_allow_html=True)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        # Market index selection
        market_options = list(st.session_state.returns_data.columns)
        btc_default = next((col for col in market_options if 'BTC' in str(col)), market_options[0])
        
        market_index = st.selectbox(
            "Select Market Index:",
            options=market_options,
            index=market_options.index(btc_default) if btc_default in market_options else 0
        )
        
        risk_free_rate = st.slider("Risk-Free Rate (Annual %)", min_value=0.0, max_value=10.0, value=3.0, step=0.1) / 100
    
    with col2:
        trading_days = st.slider("Trading Days per Year", min_value=200, max_value=365, value=252)
        frontier_points = st.slider("Frontier Points", min_value=50, max_value=300, value=120)
    
    if st.button("🚀 Build Simple Index Model"):
        returns_data = st.session_state.returns_data.dropna()
        rf_daily = (1 + risk_free_rate)**(1/trading_days) - 1
        
        # Market data
        market_returns = returns_data[market_index].values
        market_excess = market_returns - rf_daily
        market_var_daily = np.var(market_excess, ddof=0)
        
        # Assets (excluding market index)
        assets = [col for col in returns_data.columns if col != market_index]
        
        if not assets:
            st.error("❌ No assets available for SIM analysis.")
            st.stop()
        
        st.success(f"✅ Building SIM with {market_index} as market index")
        
        # SIM parameter estimation
        sim_params = []
        
        for asset in assets:
            asset_returns = returns_data[asset].values
            asset_excess = asset_returns - rf_daily
            
            # OLS regression: R_i - R_f = α_i + β_i(R_m - R_f) + ε_i
            alpha, beta, r2 = ols_manual(market_excess, asset_excess)
            
            # Calculate residual variance
            residuals = asset_excess - (alpha + beta * market_excess)
            residual_var_daily = np.var(residuals, ddof=0)
            
            # Total variance decomposition
            systematic_var = (beta**2) * market_var_daily
            total_var_daily = systematic_var + residual_var_daily
            
            sim_params.append({
                'Asset': asset,
                'Alpha_daily': alpha,
                'Beta': beta,
                'R²': r2,
                'Systematic_Var': systematic_var,
                'Residual_Var': residual_var_daily,
                'Total_Var': total_var_daily,
                'Systematic_Risk_%': (systematic_var / total_var_daily) * 100 if total_var_daily > 0 else 0
            })
        
        sim_df = pd.DataFrame(sim_params)
        
        # Display SIM parameters
        st.markdown('<h2 class="section-header">📊 SIM Parameter Estimates</h2>', unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Beta", f"{sim_df['Beta'].mean():.3f}")
        with col2:
            st.metric("Average R²", f"{sim_df['R²'].mean():.3f}")
        with col3:
            st.metric("Avg Systematic Risk %", f"{sim_df['Systematic_Risk_%'].mean():.1f}%")
        with col4:
            st.metric("Market Variance (daily)", f"{market_var_daily:.6f}")
        
        # Detailed parameters table
        st.subheader("📋 Detailed SIM Parameters")
        display_sim_df = sim_df.copy()
        numeric_cols = ['Alpha_daily', 'Beta', 'R²', 'Systematic_Var', 'Residual_Var', 'Total_Var']
        for col in numeric_cols:
            display_sim_df[col] = display_sim_df[col].round(6)
        
        st.dataframe(display_sim_df, use_container_width=True)
        
        # Risk decomposition visualization
        st.subheader("🎯 Risk Decomposition")
        
        fig_risk = px.bar(
            sim_df,
            x='Asset',
            y=['Systematic_Var', 'Residual_Var'],
            title='Risk Decomposition: Systematic vs Idiosyncratic Risk',
            labels={'value': 'Variance', 'variable': 'Risk Type'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Build SIM covariance matrix
        n_assets = len(assets)
        cov_sim_daily = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                beta_i = sim_df['Beta'].iloc[i]
                beta_j = sim_df['Beta'].iloc[j]
                
                if i == j:
                    # Diagonal: total variance
                    cov_sim_daily[i, j] = sim_df['Total_Var'].iloc[i]
                else:
                    # Off-diagonal: systematic covariance only
                    cov_sim_daily[i, j] = beta_i * beta_j * market_var_daily
        
        # Annualize for portfolio optimization
        cov_sim_annual = cov_sim_daily * trading_days
        mean_returns_annual = returns_data[assets].mean() * trading_days
        
        # SIM Portfolio Optimization
        st.markdown('<h2 class="section-header">💼 SIM Portfolio Optimization</h2>', unsafe_allow_html=True)
        
        # Analytical efficient frontier for SIM
        def markowitz_frontier_sim(mu, cov, n_points=120):
            try:
                inv_cov = np.linalg.inv(cov)
                ones = np.ones(len(mu))
                
                A = ones @ inv_cov @ ones
                B = ones @ inv_cov @ mu
                C = mu @ inv_cov @ mu
                D = A * C - B * B
                
                if D <= 0:
                    return None, None, None, None
                
                # Target returns range
                mu_min = mu.min() * 0.9
                mu_max = mu.max() * 1.1
                targets = np.linspace(mu_min, mu_max, n_points)
                
                vols, rets, weights = [], [], []
                
                for target in targets:
                    lam = (C - B * target) / D
                    gam = (A * target - B) / D
                    w = inv_cov @ (lam * ones + gam * mu)
                    
                    ret = w @ mu
                    vol = np.sqrt(w @ cov @ w)
                    
                    vols.append(vol)
                    rets.append(ret)
                    weights.append(w)
                
                return np.array(vols), np.array(rets), weights, targets
            
            except:
                return None, None, None, None
        
        # Generate SIM efficient frontier
        vols_sim, rets_sim, weights_sim, targets_sim = markowitz_frontier_sim(
            mean_returns_annual.values, cov_sim_annual, frontier_points
        )
        
        if vols_sim is not None:
            # Find key portfolios
            # Global minimum variance
            inv_cov = np.linalg.inv(cov_sim_annual)
            ones = np.ones(n_assets)
            w_gmv = inv_cov @ ones / (ones @ inv_cov @ ones)
            ret_gmv = w_gmv @ mean_returns_annual.values
            vol_gmv = np.sqrt(w_gmv @ cov_sim_annual @ w_gmv)
            
            # Maximum Sharpe ratio (tangency)
            def neg_sharpe_sim(w):
                ret = w @ mean_returns_annual.values
                vol = np.sqrt(w @ cov_sim_annual @ w)
                return -(ret - risk_free_rate) / vol if vol > 0 else 1e10
            
            from scipy.optimize import minimize
            
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
            bounds = tuple((0, 1) for _ in range(n_assets))
            x0 = np.ones(n_assets) / n_assets
            
            result_tang = minimize(neg_sharpe_sim, x0, bounds=bounds, constraints=constraints)
            
            if result_tang.success:
                w_tang = result_tang.x
                ret_tang = w_tang @ mean_returns_annual.values
                vol_tang = np.sqrt(w_tang @ cov_sim_annual @ w_tang)
                sharpe_tang = (ret_tang - risk_free_rate) / vol_tang
            else:
                w_tang = w_gmv
                ret_tang = ret_gmv
                vol_tang = vol_gmv
                sharpe_tang = (ret_tang - risk_free_rate) / vol_tang
            
            # Display key portfolios
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🛡️ SIM Global Minimum Variance")
                st.metric("Expected Return", f"{ret_gmv*100:.2f}%")
                st.metric("Volatility", f"{vol_gmv*100:.2f}%")
                
                gmv_weights_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight (%)': (w_gmv * 100).round(2)
                }).sort_values('Weight (%)', ascending=False)
                st.dataframe(gmv_weights_df, use_container_width=True)
            
            with col2:
                st.subheader("⚡ SIM Tangency Portfolio")
                st.metric("Expected Return", f"{ret_tang*100:.2f}%")
                st.metric("Volatility", f"{vol_tang*100:.2f}%")
                st.metric("Sharpe Ratio", f"{sharpe_tang:.3f}")
                
                tang_weights_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight (%)': (w_tang * 100).round(2)
                }).sort_values('Weight (%)', ascending=False)
                st.dataframe(tang_weights_df, use_container_width=True)
            
            # Plot SIM efficient frontier
            st.subheader("📈 SIM Efficient Frontier")
            
            fig_sim = go.Figure()
            
            # SIM efficient frontier
            fig_sim.add_trace(go.Scatter(
                x=vols_sim * 100,
                y=rets_sim * 100,
                mode='lines',
                name='SIM Efficient Frontier',
                line=dict(color='orange', width=3)
            ))
            
            # Individual assets
            asset_vols = np.sqrt(np.diag(cov_sim_annual)) * 100
            asset_rets = mean_returns_annual.values * 100
            
            # Ensure asset names are strings
            asset_labels = [str(asset) for asset in assets]
            fig_sim.add_trace(go.Scatter(
                x=asset_vols,
                y=asset_rets,
                mode='markers+text',
                name='Individual Assets',
                text=asset_labels,
                textposition="top center",
                marker=dict(size=10, color='red')
            ))
            
            # Key portfolios
            fig_sim.add_trace(go.Scatter(
                x=[vol_gmv * 100],
                y=[ret_gmv * 100],
                mode='markers',
                name='SIM GMV',
                marker=dict(size=15, color='green', symbol='star')
            ))
            
            fig_sim.add_trace(go.Scatter(
                x=[vol_tang * 100],
                y=[ret_tang * 100],
                mode='markers',
                name='SIM Tangency',
                marker=dict(size=15, color='gold', symbol='star')
            ))
            
            # Capital Market Line
            cml_x = np.linspace(0, vols_sim.max() * 1.1 * 100, 100)
            cml_y = risk_free_rate * 100 + sharpe_tang * cml_x
            fig_sim.add_trace(go.Scatter(
                x=cml_x,
                y=cml_y,
                mode='lines',
                name='SIM CML',
                line=dict(color='green', dash='dash', width=2)
            ))
            
            fig_sim.update_layout(
                title='Simple Index Model - Efficient Frontier',
                xaxis_title='Volatility (%)',
                yaxis_title='Expected Return (%)',
                height=600
            )
            
            st.plotly_chart(fig_sim, use_container_width=True)
            
            # Compare with full Markowitz (if available)
            if 'portfolio_results' in st.session_state and 'markowitz' in st.session_state.portfolio_results:
                st.subheader("⚖️ SIM vs Markowitz Comparison")
                
                markowitz_data = st.session_state.portfolio_results['markowitz']
                
                fig_compare = go.Figure()
                
                # Markowitz frontier
                fig_compare.add_trace(go.Scatter(
                    x=markowitz_data['efficient_frontier']['Volatility'] * 100,
                    y=markowitz_data['efficient_frontier']['Return'] * 100,
                    mode='lines',
                    name='Markowitz Frontier',
                    line=dict(color='blue', width=2)
                ))
                
                # SIM frontier
                fig_compare.add_trace(go.Scatter(
                    x=vols_sim * 100,
                    y=rets_sim * 100,
                    mode='lines',
                    name='SIM Frontier',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                fig_compare.update_layout(
                    title='Efficient Frontier Comparison: Markowitz vs SIM',
                    xaxis_title='Volatility (%)',
                    yaxis_title='Expected Return (%)',
                    height=500
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Correlation analysis
                if len(vols_sim) == len(markowitz_data['efficient_frontier']):
                    corr_returns = np.corrcoef(rets_sim, markowitz_data['efficient_frontier']['Return'])[0, 1]
                    corr_vols = np.corrcoef(vols_sim, markowitz_data['efficient_frontier']['Volatility'])[0, 1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Returns Correlation", f"{corr_returns:.4f}")
                    with col2:
                        st.metric("Volatility Correlation", f"{corr_vols:.4f}")
            
            # Store SIM results
            if 'sim_results' not in st.session_state:
                st.session_state.sim_results = {}
            
            st.session_state.sim_results = {
                'parameters': sim_df,
                'covariance_matrix': cov_sim_annual,
                'mean_returns': mean_returns_annual,
                'efficient_frontier': {
                    'volatilities': vols_sim,
                    'returns': rets_sim,
                    'weights': weights_sim
                },
                'gmv_portfolio': {
                    'weights': w_gmv,
                    'return': ret_gmv,
                    'volatility': vol_gmv
                },
                'tangency_portfolio': {
                    'weights': w_tang,
                    'return': ret_tang,
                    'volatility': vol_tang,
                    'sharpe_ratio': sharpe_tang
                },
                'market_index': market_index
            }
        
        else:
            st.error("❌ Failed to generate SIM efficient frontier. Check data quality.")

# ===============================================================
# PAGE 7: MODEL COMPARISON
# ===============================================================

elif page == "⚖️ Model Comparison":
    st.markdown('<h1 class="main-header">Comprehensive Model Comparison</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page.")
        st.stop()
    
    # Check which models have been run
    models_available = []
    if 'portfolio_results' in st.session_state and 'markowitz' in st.session_state.portfolio_results:
        models_available.append('Markowitz')
    if 'capm_results' in st.session_state:
        models_available.append('CAPM')
    if 'sim_results' in st.session_state:
        models_available.append('SIM')
    if 'arima_results' in st.session_state:
        models_available.append('ARIMA')
    
    if not models_available:
        st.warning("⚠️ No models have been run yet. Please run analyses from other pages first.")
        st.info("Available analyses: Portfolio Optimization, CAPM Analysis, Simple Index Model, ARIMA Modeling")
        st.stop()
    
    st.success(f"✅ Available models for comparison: {', '.join(models_available)}")
    
    # ===============================================================
    # PORTFOLIO MODELS COMPARISON (Markowitz vs SIM)
    # ===============================================================
    
    if 'Markowitz' in models_available and 'SIM' in models_available:
        st.markdown('<h2 class="section-header">💼 Portfolio Models Comparison</h2>', unsafe_allow_html=True)
        
        markowitz_data = st.session_state.portfolio_results['markowitz']
        sim_data = st.session_state.sim_results
        
        # Efficient Frontiers Comparison
        st.subheader("📈 Efficient Frontiers Overlay")
        
        fig_frontiers = go.Figure()
        
        # Markowitz frontier
        fig_frontiers.add_trace(go.Scatter(
            x=markowitz_data['efficient_frontier']['Volatility'] * 100,
            y=markowitz_data['efficient_frontier']['Return'] * 100,
            mode='lines',
            name='Markowitz (Full Covariance)',
            line=dict(color='blue', width=3)
        ))
        
        # SIM frontier
        fig_frontiers.add_trace(go.Scatter(
            x=sim_data['efficient_frontier']['volatilities'] * 100,
            y=sim_data['efficient_frontier']['returns'] * 100,
            mode='lines',
            name='Simple Index Model',
            line=dict(color='orange', width=3, dash='dash')
        ))
        
        # Key portfolios
        fig_frontiers.add_trace(go.Scatter(
            x=[markowitz_data['max_sharpe_portfolio']['volatility'] * 100],
            y=[markowitz_data['max_sharpe_portfolio']['return'] * 100],
            mode='markers',
            name='Markowitz Tangency',
            marker=dict(size=15, color='blue', symbol='star')
        ))
        
        fig_frontiers.add_trace(go.Scatter(
            x=[sim_data['tangency_portfolio']['volatility'] * 100],
            y=[sim_data['tangency_portfolio']['return'] * 100],
            mode='markers',
            name='SIM Tangency',
            marker=dict(size=15, color='orange', symbol='star')
        ))
        
        fig_frontiers.update_layout(
            title='Efficient Frontiers: Markowitz vs Simple Index Model',
            xaxis_title='Volatility (%)',
            yaxis_title='Expected Return (%)',
            height=600
        )
        
        st.plotly_chart(fig_frontiers, use_container_width=True)
        
        # Quantitative Comparison
        st.subheader("📊 Quantitative Comparison")
        
        # Calculate frontier correlation if possible
        try:
            # Align frontiers for comparison
            min_len = min(len(markowitz_data['efficient_frontier']), len(sim_data['efficient_frontier']['returns']))
            
            mk_returns = markowitz_data['efficient_frontier']['Return'][:min_len]
            sim_returns = sim_data['efficient_frontier']['returns'][:min_len]
            
            mk_vols = markowitz_data['efficient_frontier']['Volatility'][:min_len]
            sim_vols = sim_data['efficient_frontier']['volatilities'][:min_len]
            
            corr_returns = np.corrcoef(mk_returns, sim_returns)[0, 1]
            corr_vols = np.corrcoef(mk_vols, sim_vols)[0, 1]
            
            rmse_returns = np.sqrt(np.mean((mk_returns - sim_returns) ** 2))
            rmse_vols = np.sqrt(np.mean((mk_vols - sim_vols) ** 2))
            
        except:
            corr_returns = np.nan
            corr_vols = np.nan
            rmse_returns = np.nan
            rmse_vols = np.nan
        
        # Comparison metrics
        comparison_data = {
            'Metric': [
                'Max Sharpe Return (%)',
                'Max Sharpe Volatility (%)',
                'Max Sharpe Ratio',
                'Min Vol Return (%)',
                'Min Vol Volatility (%)',
                'Frontier Returns Correlation',
                'Frontier Volatility Correlation',
                'RMSE Returns',
                'RMSE Volatility'
            ],
            'Markowitz': [
                f"{markowitz_data['max_sharpe_portfolio']['return']*100:.2f}",
                f"{markowitz_data['max_sharpe_portfolio']['volatility']*100:.2f}",
                f"{markowitz_data['max_sharpe_portfolio']['sharpe_ratio']:.3f}",
                f"{markowitz_data['min_vol_portfolio']['return']*100:.2f}",
                f"{markowitz_data['min_vol_portfolio']['volatility']*100:.2f}",
                f"{corr_returns:.4f}" if not np.isnan(corr_returns) else "N/A",
                f"{corr_vols:.4f}" if not np.isnan(corr_vols) else "N/A",
                f"{rmse_returns:.6f}" if not np.isnan(rmse_returns) else "N/A",
                f"{rmse_vols:.6f}" if not np.isnan(rmse_vols) else "N/A"
            ],
            'SIM': [
                f"{sim_data['tangency_portfolio']['return']*100:.2f}",
                f"{sim_data['tangency_portfolio']['volatility']*100:.2f}",
                f"{sim_data['tangency_portfolio']['sharpe_ratio']:.3f}",
                f"{sim_data['gmv_portfolio']['return']*100:.2f}",
                f"{sim_data['gmv_portfolio']['volatility']*100:.2f}",
                "Baseline",
                "Baseline",
                "Baseline",
                "Baseline"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    # ===============================================================
    # CAPM EVALUATION
    # ===============================================================
    
    if 'CAPM' in models_available:
        st.markdown('<h2 class="section-header">📊 CAPM Model Evaluation</h2>', unsafe_allow_html=True)
        
        capm_data = st.session_state.capm_results['results_df']
        
        # CAPM Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_r2 = capm_data['R²'].mean()
            st.metric("Average R²", f"{avg_r2:.3f}")
        
        with col2:
            significant_betas = (capm_data['p_Beta'] < 0.05).sum()
            st.metric("Significant Betas", f"{significant_betas}/{len(capm_data)}")
        
        with col3:
            avg_rmse = capm_data['RMSE'].mean()
            st.metric("Average RMSE", f"{avg_rmse:.6f}")
        
        with col4:
            avg_mae = capm_data['MAE'].mean()
            st.metric("Average MAE", f"{avg_mae:.6f}")
        
        # CAPM vs Actual Returns
        st.subheader("🎯 CAPM Prediction Accuracy")
        
        fig_capm_accuracy = px.scatter(
            capm_data,
            x='Ann_Return (%)',
            y='CAPM_Predicted (%)',
            size='R²',
            hover_data=['Asset', 'Beta'],
            title='CAPM: Actual vs Predicted Returns'
        )
        
        # Perfect prediction line
        min_ret = min(capm_data['Ann_Return (%)'].min(), capm_data['CAPM_Predicted (%)'].min())
        max_ret = max(capm_data['Ann_Return (%)'].max(), capm_data['CAPM_Predicted (%)'].max())
        
        fig_capm_accuracy.add_trace(
            go.Scatter(
                x=[min_ret, max_ret],
                y=[min_ret, max_ret],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red', width=2)
            )
        )
        
        st.plotly_chart(fig_capm_accuracy, use_container_width=True)
        
        # Beta Distribution
        st.subheader("📊 Beta Distribution Analysis")
        
        fig_beta_dist = px.histogram(
            capm_data,
            x='Beta',
            nbins=20,
            title='Distribution of Beta Coefficients'
        )
        fig_beta_dist.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Market Beta = 1")
        st.plotly_chart(fig_beta_dist, use_container_width=True)
    
    # ===============================================================
    # ARIMA MODEL EVALUATION
    # ===============================================================
    
    if 'ARIMA' in models_available:
        st.markdown('<h2 class="section-header">📈 ARIMA Models Summary</h2>', unsafe_allow_html=True)
        
        arima_data = st.session_state.arima_results
        
        # ARIMA Results Summary
        arima_summary = []
        for asset, results in arima_data.items():
            arima_summary.append({
                'Asset': asset,
                'ARIMA Order': f"({results['p']},0,{results['q']})",
                'AIC': results['model']['aic'],
                'BIC': results['model']['bic'],
                'Log-Likelihood': -results['model']['nll'],
                'Residual Variance': results['model']['sigma2'],
                'RMSE': results.get('rmse', 'N/A'),
                'MAE': results.get('mae', 'N/A')
            })
        
        arima_summary_df = pd.DataFrame(arima_summary)
        st.dataframe(arima_summary_df, use_container_width=True)
        
        # ARIMA Performance Visualization
        if len(arima_summary) > 1:
            fig_arima_perf = px.scatter(
                arima_summary_df,
                x='AIC',
                y='BIC',
                hover_data=['Asset', 'ARIMA Order'],
                title='ARIMA Model Performance: AIC vs BIC'
            )
            st.plotly_chart(fig_arima_perf, use_container_width=True)
    
    # ===============================================================
    # COMPREHENSIVE MODEL COMPARISON
    # ===============================================================
    
    st.markdown('<h2 class="section-header">🏆 Overall Model Assessment</h2>', unsafe_allow_html=True)
    
    # Model Complexity vs Performance
    model_assessment = []
    
    if 'Markowitz' in models_available:
        model_assessment.append({
            'Model': 'Markowitz Portfolio Theory',
            'Complexity': 'High',
            'Data Requirements': 'Full covariance matrix',
            'Key Strength': 'Optimal risk-return trade-off',
            'Key Limitation': 'Estimation error sensitivity',
            'Best Use Case': 'Theoretical benchmark, small portfolios'
        })
    
    if 'SIM' in models_available:
        model_assessment.append({
            'Model': 'Simple Index Model',
            'Complexity': 'Medium',
            'Data Requirements': 'Market index + individual assets',
            'Key Strength': 'Reduced parameter estimation',
            'Key Limitation': 'Single factor assumption',
            'Best Use Case': 'Large portfolios, factor-based investing'
        })
    
    if 'CAPM' in models_available:
        avg_r2 = capm_data['R²'].mean()
        model_assessment.append({
            'Model': 'Capital Asset Pricing Model',
            'Complexity': 'Low',
            'Data Requirements': 'Market proxy + risk-free rate',
            'Key Strength': 'Simple pricing framework',
            'Key Limitation': f'Low explanatory power (R²={avg_r2:.2f})',
            'Best Use Case': 'Asset pricing, cost of capital estimation'
        })
    
    if 'ARIMA' in models_available:
        avg_aic = np.mean([results['model']['aic'] for results in arima_data.values()])
        model_assessment.append({
            'Model': 'ARIMA Time Series',
            'Complexity': 'Medium-High',
            'Data Requirements': 'Historical time series',
            'Key Strength': 'Captures temporal dependencies',
            'Key Limitation': 'Assumes stationarity',
            'Best Use Case': 'Short-term forecasting, volatility modeling'
        })
    
    assessment_df = pd.DataFrame(model_assessment)
    st.dataframe(assessment_df, use_container_width=True)
    
    # Performance Summary Chart
    if len(models_available) > 1:
        st.subheader("📊 Model Performance Summary")
        
        # Create performance metrics
        performance_metrics = {
            'Model': [],
            'Complexity Score': [],
            'Accuracy Score': [],
            'Practical Score': []
        }
        
        for model in models_available:
            performance_metrics['Model'].append(model)
            
            if model == 'Markowitz':
                performance_metrics['Complexity Score'].append(9)  # High complexity
                performance_metrics['Accuracy Score'].append(10)  # Theoretical optimum
                performance_metrics['Practical Score'].append(6)   # Limited by estimation error
            
            elif model == 'SIM':
                performance_metrics['Complexity Score'].append(6)  # Medium complexity
                performance_metrics['Accuracy Score'].append(8)   # Good approximation
                performance_metrics['Practical Score'].append(8)  # Good practical performance
            
            elif model == 'CAPM':
                performance_metrics['Complexity Score'].append(3)  # Low complexity
                performance_metrics['Accuracy Score'].append(4)   # Poor fit for crypto
                performance_metrics['Practical Score'].append(7)  # Simple to implement
            
            elif model == 'ARIMA':
                performance_metrics['Complexity Score'].append(7)  # Medium-high complexity
                performance_metrics['Accuracy Score'].append(6)   # Moderate forecasting accuracy
                performance_metrics['Practical Score'].append(7)  # Good for short-term forecasting
        
        perf_df = pd.DataFrame(performance_metrics)
        
        fig_radar = go.Figure()
        
        for i, model in enumerate(perf_df['Model']):
            fig_radar.add_trace(go.Scatterpolar(
                r=[perf_df.iloc[i]['Complexity Score'], 
                   perf_df.iloc[i]['Accuracy Score'], 
                   perf_df.iloc[i]['Practical Score']],
                theta=['Complexity', 'Accuracy', 'Practicality'],
                fill='toself',
                name=model
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Final Recommendations
    st.markdown('<h2 class="section-header">💡 Recommendations</h2>', unsafe_allow_html=True)
    
    recommendations = """
    ### 🎯 Model Selection Guidelines:
    
    **For Portfolio Optimization:**
    - **Small portfolios (< 10 assets)**: Use Markowitz for theoretical optimum
    - **Large portfolios (> 20 assets)**: Use Simple Index Model for stability
    - **Factor-based strategies**: SIM with appropriate market index
    
    **For Asset Pricing:**
    - **Traditional assets**: CAPM may provide reasonable estimates
    - **Cryptocurrency assets**: CAPM shows poor fit; consider multi-factor models
    - **Risk assessment**: Use beta from CAPM with caution for crypto
    
    **For Forecasting:**
    - **Short-term (1-5 days)**: ARIMA models for return prediction
    - **Volatility forecasting**: GARCH extensions of ARIMA
    - **Long-term**: Fundamental analysis over time series models
    
    **For Risk Management:**
    - **Diversification**: Use correlation analysis from Markowitz framework
    - **Systematic risk**: Beta estimates from CAPM/SIM
    - **Idiosyncratic risk**: Residual variance from SIM decomposition
    """
    
    st.markdown(recommendations)

# ===============================================================
# MAIN APP EXECUTION
# ===============================================================

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        <p>Mathematical Modeling in Finance | Comprehensive Analysis Platform</p>
        <p>Implemented Models: ARIMA • Portfolio Optimization • CAPM • Simple Index Model</p>
    </div>
    """, 
    unsafe_allow_html=True
)
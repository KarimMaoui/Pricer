import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Options Strategy Pro", layout="wide")

# --- FONCTIONS MATHEMATIQUES ---

def black_scholes(S, K, T, r, sigma, q, option_type="Call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def get_greeks(S, K, T, r, sigma, q, option_type="Call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    mult = 1 if option_type == "Call" else -1
    
    delta = np.exp(-q * T) * (norm.cdf(d1) if option_type == "Call" else -norm.cdf(-d1))
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100
    # Simplified theta for display
    theta = (- (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))) / 365
    return delta, gamma, theta, vega

# --- LOGIQUE DES STRATEGIES ---

def get_strategy_data(strategy, S, K, T, r, sigma, q, position="Long"):
    # Structure: (Type, Strike, Quantity) - Quantity < 0 pour Vente
    pos_mult = 1 if position == "Long" else -1
    
    legs = []
    if strategy == "Call":
        legs = [("Call", K, 1 * pos_mult)]
    elif strategy == "Put":
        legs = [("Put", K, 1 * pos_mult)]
    elif strategy == "Straddle":
        legs = [("Call", K, 1 * pos_mult), ("Put", K, 1 * pos_mult)]
    elif strategy == "Strangle":
        legs = [("Call", K * 1.1, 1 * pos_mult), ("Put", K * 0.9, 1 * pos_mult)]
    elif strategy == "Bull Call Spread":
        legs = [("Call", K, 1 * pos_mult), ("Call", K * 1.1, -1 * pos_mult)]
    elif strategy == "Bear Put Spread":
        legs = [("Put", K, 1 * pos_mult), ("Put", K * 0.9, -1 * pos_mult)]
    elif strategy == "Iron Condor":
        legs = [("Put", K*0.85, 1*pos_mult), ("Put", K*0.9, -1*pos_mult), 
                ("Call", K*1.1, -1*pos_mult), ("Call", K*1.15, 1*pos_mult)]
    elif strategy == "Butterfly":
        legs = [("Call", K*0.9, 1*pos_mult), ("Call", K, -2*pos_mult), ("Call", K*1.1, 1*pos_mult)]

    total_price, total_delta, total_gamma, total_theta, total_vega = 0, 0, 0, 0, 0
    
    for opt_type, k_leg, qty in legs:
        p = black_scholes(S, k_leg, T, r, sigma, q, opt_type)
        d, g, t, v = get_greeks(S, k_leg, T, r, sigma, q, opt_type)
        total_price += p * qty
        total_delta += d * qty
        total_gamma += g * qty
        total_theta += t * qty
        total_vega += v * qty
        
    return total_price, total_delta, total_gamma, total_theta, total_vega, legs

# --- INTERFACE ---

st.title("🛡️ Options Strategy Simulator")

col1, col2 = st.columns([1, 3])

with col1:
    with st.container(border=True):
        st.subheader("Config")
        strat = st.selectbox("Stratégie", ["Call", "Put", "Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", "Iron Condor", "Butterfly"])
        side = st.pills("Position", ["Long", "Short"], default="Long")
        
        S = st.number_input("Spot Price", value=100.0)
        K = st.number_input("Strike (Base)", value=100.0)
        T = st.slider("Maturity (Years)", 0.01, 2.0, 0.5)
        sigma = st.slider("Volatility", 0.05, 1.0, 0.2)
        r = st.number_input("Rate", value=0.03)

price, delta, gamma, theta, vega, legs = get_strategy_data(strat, S, K, T, r, sigma, 0.0, side)

with col2:
    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cost/Premium", f"{price:.2f}")
    m2.metric("Delta", f"{delta:.2f}")
    m3.metric("Gamma", f"{gamma:.4f}")
    m4.metric("Theta (day)", f"{theta:.3f}")
    m5.metric("Vega", f"{vega:.2f}")

    # Plot
    st.subheader("Payoff à l'échéance")
    S_range = np.linspace(S * 0.5, S * 1.5, 100)
    
    payoff_maturity = np.zeros_like(S_range)
    for opt_type, k_leg, qty in legs:
        if opt_type == "Call":
            payoff_maturity += np.maximum(S_range - k_leg, 0) * qty
        else:
            payoff_maturity += np.maximum(k_leg - S_range, 0) * qty
    
    # On soustrait/ajoute le coût initial pour avoir le Profit/Loss (PnL)
    pnl = payoff_maturity - price

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(S_range, pnl, color="#00FFAA", linewidth=2, label="PnL at Expiry")
    ax.fill_between(S_range, pnl, 0, where=(pnl > 0), color='green', alpha=0.3)
    ax.fill_between(S_range, pnl, 0, where=(pnl < 0), color='red', alpha=0.3)
    ax.axhline(0, color='white', linewidth=1)
    ax.axvline(S, color='orange', linestyle='--', label=f"Current Spot: {S}")
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.legend()
    st.pyplot(fig)

    with st.expander("Composition de la stratégie"):
        for opt_type, k_leg, qty in legs:
            action = "Achat" if qty > 0 else "Vente"
            st.write(f"- **{action}** de {abs(qty)} **{opt_type}** strike **{k_leg:.1f}**")


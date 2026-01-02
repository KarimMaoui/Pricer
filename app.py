# streamlit run /Users/valentinfaure/Documents/Academique/SKEMA/M2\ -\ FMI/Cours/Python/Pricer\ project/streamlit_app.py

import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pricer", layout="wide")


def compute_d1(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1


def compute_d2(d1, sigma, T):
    d2 = d1 - sigma * np.sqrt(T)
    return d2


def compute_black_scholes_price(option_type, S, K, T, r, sigma, q):
    d1 = compute_d1(S, K, T, r, sigma, q)
    d2 = compute_d2(d1, sigma, T)
    
    if option_type=="Call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type=="Put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
    return price


def compute_greeks(option_type, S, K, T, r, sigma, q):
    d1 = compute_d1(S, K, T, r, sigma, q)
    d2 = compute_d2(d1, sigma, T)
    
    if option_type=="Call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = ( - (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1) ) / 365
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        
    elif option_type=="Put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = ( - (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)
                  - q * S * np.exp(-q * T) * norm.cdf(-d1) ) / 365
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
    return delta, gamma, theta, vega, rho


def plot_payoff(option_type, S, K):
    S_T = np.linspace(0.5 * S, 1.5 * S, 200)  # 50% à 150% du spot

    if option_type == "Call":
        payoff = np.maximum(S_T - K, 0)
    elif option_type == "Put":
        payoff = np.maximum(K - S_T, 0)
    else:
        st.error("Type d'option invalide")
        return

    fig, ax = plt.subplots()
    ax.plot(S_T, payoff, label=f"{option_type} Payoff")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(S, color='red', linestyle='--', label="Spot S")
    ax.set_xlabel("Price at Maturity S_T")
    ax.set_ylabel("Payoff")
    ax.set_title(f"{option_type} Option Payoff at Maturity")
    ax.legend()
    ax.grid(True)
    return fig



col1, col2 = st.columns([1,3])

with col1:
    with st.container(border=True):
        st.markdown("#### Parameters")
        option_type= st.pills("Option type", ["Call","Put"], default="Call")
        S = st.number_input("Spot price", value=100, min_value=0)
        K = st.number_input("Strike", value=100, min_value=0)
        T = st.number_input("Time to maturity (in years)", value=1.0, min_value=0.001)
        r = st.number_input("Risk-free rate", value=0.05)
        sigma = st.number_input("Volatility", value=0.2, min_value=0.001)
        q = st.number_input("Dividend yield", value=0.02)

price = compute_black_scholes_price(option_type, S, K, T, r, sigma, q)
delta, gamma, theta, vega, rho = compute_greeks(option_type, S, K, T, r, sigma, q)
fig = plot_payoff(option_type, S, K)


with col2:
    
    col1,col2,col3,col4,col5,col6 = st.columns(6)
    with col1:
        with st.container(border=True):
            st.metric(label="Price", value=round(price,2))
    with col2:
        with st.container(border=True):
            st.metric(label="Delta", value=round(delta,2))
    with col3:
        with st.container(border=True):
            st.metric(label="Gamma", value=round(gamma,2))
    with col4:
        with st.container(border=True):
            st.metric(label="Theta", value=round(theta,2))
    with col5:
        with st.container(border=True):
            st.metric(label="Vega", value=round(vega,2))
    with col6:
        with st.container(border=True):
            st.metric(label="Rho", value=round(rho,2))
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Payoff","Delta", "Gamma", "Theta", "Vega", "Rho"])
    
    with tab1:
        st.pyplot(fig)


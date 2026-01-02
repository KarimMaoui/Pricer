import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Options Strategy Pro", layout="wide")

# --- 1. MOTEUR MATHÉMATIQUE ---

def black_scholes(S, K, T, r, sigma, q, option_type="Call"):
    if option_type == "Stock":
        return S 

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def get_greeks(S, K, T, r, sigma, q, option_type="Call"):
    if option_type == "Stock":
        return 1.0, 0.0, 0.0, 0.0 

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = np.exp(-q * T) * (norm.cdf(d1) if option_type == "Call" else -norm.cdf(-d1))
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100
    theta = (- (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))) / 365
    return delta, gamma, theta, vega

# --- 2. LOGIQUE DES STRATÉGIES ---

def get_strategy_description(strategy, position):
    desc = {
        "Call": {
            "Long": "Achat d'un droit d'acheter. **Haussier**. Vous pariez sur une forte hausse.",
            "Short": "Vente d'un droit d'acheter (Naked Call). **Baissier**. Très risqué."
        },
        "Put": {
            "Long": "Achat d'un droit de vendre. **Baissier**. Vous pariez sur une baisse.",
            "Short": "Vente d'un droit de vendre. **Neutre/Haussier**."
        },
        "Covered Call": {
            "Long": "**Revenu**. Long Stock + Short Call. Pour marché neutre/légèrement haussier.",
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": "**Assurance**. Long Stock + Long Put. Protection contre la baisse.",
            "Short": "N/A"
        },
        "Straddle": {
            "Long": "**Volatilité Pure**. Achat Call + Achat Put (même strike). Idéal avant une annonce majeure.",
            "Short": "**Vente de Volatilité**. Gain si le marché ne bouge pas."
        },
        "Strangle": {
            "Long": "Comme le Straddle mais moins cher (Strikes écartés). Nécessite un mouvement violent.",
            "Short": "Vente de volatilité avec marge d'erreur."
        },
        "Bull Call Spread": {
            "Long": "**Modérément Haussier**. Gain plafonné, coût réduit.",
            "Short": "**Modérément Baissier**. Encaissement de crédit."
        },
        "Bear Put Spread": {
            "Long": "**Modérément Baissier**. Gain plafonné, coût réduit.",
            "Short": "**Modérément Haussier**. Encaissement de crédit."
        },
        "Butterfly": {
            "Long": "**Cible Précise**. Pari sur une stagnation exacte au strike central.",
            "Short": "Pari sur une sortie de la zone centrale."
        },
        "Call Ratio Backspread": {
            "Long": "**Explosion Haussière (Volatilité)**. Vous vendez 1 Call ATM pour financer l'achat de 2 Calls OTM. <br>✅ **Gain illimité** si le marché explose.<br>✅ **Risque limité** (voire gain) si le marché s'effondre.<br>❌ **Perte max** si le marché monte doucement et stagne au strike haut.<br>👉 *Très utilisé sur les matières premières (Pétrole, Agricole).* ",
            "Short": "Rare. Pari sur une hausse modérée et calme."
        }
    }
    return desc.get(strategy, {}).get(position, "Description non disponible.")

def get_strategy_legs(strategy, K, position="Long"):
    pos_mult = 1 if position == "Long" else -1
    
    if strategy == "Call":
        return [("Call", 1.0, 1 * pos_mult)]
    elif strategy == "Put":
        return [("Put", 1.0, 1 * pos_mult)]
    elif strategy == "Covered Call":
        return [("Stock", 0, 1), ("Call", 1.1, -1)] 
    elif strategy == "Protective Put":
        return [("Stock", 0, 1), ("Put", 0.9, 1)] 
    elif strategy == "Straddle":
        return [("Call", 1.0, 1 * pos_mult), ("Put", 1.0, 1 * pos_mult)]
    elif strategy == "Strangle":
        return [("Call", 1.1, 1 * pos_mult), ("Put", 0.9, 1 * pos_mult)]
    elif strategy == "Bull Call Spread":
        return [("Call", 1.0, 1 * pos_mult), ("Call", 1.1, -1 * pos_mult)]
    elif strategy == "Bear Put Spread":
        return [("Put", 1.0, 1 * pos_mult), ("Put", 0.9, -1 * pos_mult)]
    elif strategy == "Butterfly":
        return [("Call", 0.9, 1*pos_mult), ("Call", 1.0, -2*pos_mult), ("Call", 1.1, 1*pos_mult)]
    
    # NOUVELLE STRATEGIE AJOUTÉE
    elif strategy == "Call Ratio Backspread":
        # Ratio 1x2 : On vend 1 Call ATM, on achète 2 Calls OTM
        # Si Position Long = On fait le montage standard
        return [("Call", 1.0, -1 * pos_mult), ("Call", 1.2, 2 * pos_mult)]
    
    return []

# --- 3. INTERFACE ---

st.title("🎓 Master Options Strategies")
st.markdown("Simulateur de PnL et Analyseur de Grecques")

col_params, col_viz = st.columns([1, 3])

with col_params:
    with st.container(border=True):
        st.header("1. Stratégie")
        # Ajout de la nouvelle stratégie dans la liste
        strat_list = ["Call", "Put", "Covered Call", "Protective Put", 
                      "Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", 
                      "Butterfly", "Call Ratio Backspread"]
        
        selected_strat = st.selectbox("Type de montage", strat_list, index=9) # Index 9 pour sélectionner la nouvelle par défaut
        
        if selected_strat in ["Covered Call", "Protective Put"]:
            position = "Long"
            st.info(f"Position fixée à 'Long' pour {selected_strat}")
        else:
            position = st.pills("Votre coté", ["Long", "Short"], default="Long")

        st.divider()
        st.header("2. Marché")
        S = st.number_input("Prix Spot (S)", value=100.0)
        K = st.number_input("Strike Central (K)", value=100.0)
        T = st.slider("Maturité (Années)", 0.01, 2.0, 0.5, step=0.01) # T augmenté par défaut pour mieux voir le backspread
        sigma = st.slider("Volatilité Implicite (σ)", 0.05, 1.50, 0.30) # Vol augmentée pour l'exemple
        r = st.number_input("Taux sans risque", value=0.04)

# Calculs
legs_config = get_strategy_legs(selected_strat, K, position)
total_price, total_delta, total_gamma, total_theta, total_vega = 0, 0, 0, 0, 0
real_legs_details = []

for leg_type, strike_mult, qty in legs_config:
    leg_k = K * strike_mult if leg_type != "Stock" else 0
    
    p = black_scholes(S, leg_k, T, r, sigma, 0, leg_type)
    d, g, t, v = get_greeks(S, leg_k, T, r, sigma, 0, leg_type)
    
    total_price += p * qty
    total_delta += d * qty
    total_gamma += g * qty
    total_theta += t * qty
    total_vega += v * qty
    
    real_legs_details.append((leg_type, leg_k, qty))

with col_viz:
    with st.expander("📖 Comprendre cette stratégie", expanded=True):
        st.markdown(f"### {selected_strat} ({position})")
        st.markdown(get_strategy_description(selected_strat, position))

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    cost_label = "Coût (Débit)" if total_price > 0 else "Crédit Reçu"
    
    kpi1.metric(cost_label, f"{abs(total_price):.2f} $", delta="-Payé" if total_price > 0 else "+Reçu", delta_color="inverse")
    kpi2.metric("Delta", f"{total_delta:.2f}")
    kpi3.metric("Gamma", f"{total_gamma:.3f}")
    kpi4.metric("Theta", f"{total_theta:.3f}")
    kpi5.metric("Vega", f"{total_vega:.2f}")

    st.subheader("Profit & Loss à l'échéance")
    
    # On élargit le range du graphique pour bien voir l'effet explosif du Backspread
    S_range = np.linspace(S * 0.5, S * 1.8, 200)
    pnl_maturity = np.zeros_like(S_range) - total_price 
    
    for leg_type, leg_k, qty in real_legs_details:
        if leg_type == "Call":
            pnl_maturity += np.maximum(S_range - leg_k, 0) * qty
        elif leg_type == "Put":
            pnl_maturity += np.maximum(leg_k - S_range, 0) * qty
        elif leg_type == "Stock":
            pnl_maturity += (S_range - S) * qty

    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity >= 0), color='green', alpha=0.2, interpolate=True)
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity < 0), color='red', alpha=0.2, interpolate=True)
    
    ax.plot(S_range, pnl_maturity, color="white", linewidth=2)
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(S, color='yellow', linestyle='--', label=f"Spot Actuel: {S}")
    
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel("Prix du Sous-jacent à Maturité", color='white')
    ax.set_ylabel("P&L ($)", color='white')
    ax.legend()
    ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    st.pyplot(fig)

    st.caption("Détail des positions (Jambes) composant la stratégie")
    legs_data = [{"Type": t, "Strike": f"{k:.2f}" if k > 0 else "Mkt", "Qté": q, "Action": "Achat" if q > 0 else "Vente"} for t, k, q in real_legs_details]
    st.dataframe(legs_data, use_container_width=True)

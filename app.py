import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Derivatives Pricer", layout="wide")

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
    def format_desc(structure, these, contexte):
        return f"""
        Structure Produit : {structure}
        
        Thèse d'investissement : {these}
        
        Contexte Marché : {contexte}
        """

    desc = {
        "Call": {
            "Long": format_desc("Directionnel Haussier.", "Levier pur sur la hausse.", "Momentum / Catalyseur."),
            "Short": format_desc("Vente à découvert.", "Encaissement de prime, risque illimité.", "Baissier / Surévaluation.")
        },
        "Put": {
            "Long": format_desc("Directionnel Baissier / Protection.", "Profit sur la baisse ou couverture.", "Correction / Hedge."),
            "Short": format_desc("Génération de Rendement (Target Buying).", "Encaissement de prime, volonté d'acheter bas.", "Neutre / Haussier.")
        },
        "Covered Call": {
            "Long": format_desc("Yield Enhancement (Long Stock + Short Call).", "Monétisation de la détention d'actifs. Gain plafonné.", "Marché neutre / Léger haussier."),
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": format_desc("Assurance Portefeuille (Long Stock + Long Put).", "Préservation du capital avec potentiel haussier intact.", "Incertitude / Earnings."),
            "Short": "N/A"
        },
        "Straddle": {
            "Long": format_desc("Volatilité Pure (ATM).", "Explosion du prix indifférente de la direction.", "Événement binaire."),
            "Short": format_desc("Vente Volatilité.", "Pari sur la stabilité des cours.", "Range trading.")
        },
        "Strangle": {
            "Long": format_desc("Volatilité Low-Cost (OTM).", "Mouvement violent requis.", "Black Swan."),
            "Short": format_desc("Vente Volatilité Large.", "Marge d'erreur plus importante.", "Range large.")
        },
        "Bull Call Spread": {
            "Long": format_desc("Haussier Risque Défini.", "Moins cher qu'un Call sec. Gain capé.", "Hausse modérée."),
            "Short": format_desc("Crédit Baissier.", "Encaissement de crédit.", "Baissier / Résistance.")
        },
        "Bear Put Spread": {
            "Long": format_desc("Baissier Risque Défini.", "Moins cher qu'un Put sec. Gain capé.", "Baisse modérée."),
            "Short": format_desc("Crédit Haussier.", "Encaissement de crédit.", "Haussier / Support.")
        },
        "Butterfly": {
            "Long": format_desc("Neutre (Target précis).", "Capture maximale de la valeur temps.", "Volatilité en baisse."),
            "Short": format_desc("Volatilité.", "Sortie de la zone centrale.", "Breakout.")
        },
        "Call Ratio Backspread": {
            "Long": format_desc(
                "1 Short ATM / 2 Long OTM.",
                "Gain illimité à la hausse. Coût d'entrée souvent nul ou négatif.",
                "Volatilité extrême haussière (Commodities/Tech)."
            ),
            "Short": format_desc("Contrarian.", "Pari risqué contre la hausse.", "Non standard.")
        }
    }
    return desc.get(strategy, {}).get(position, "N/A")

def get_strategy_legs(strategy, K, width_lower, width_upper, position="Long"):
    pos_mult = 1 if position == "Long" else -1
    
    # Stratégies simples (Call/Put/Straddle) : les sliders n'ont pas d'impact
    if strategy == "Call":
        return [("Call", 1.0, 1 * pos_mult)]
    elif strategy == "Put":
        return [("Put", 1.0, 1 * pos_mult)]
    elif strategy == "Straddle":
        return [("Call", 1.0, 1 * pos_mult), ("Put", 1.0, 1 * pos_mult)]
        
    # Stratégies utilisant la borne HAUTE (Call OTM)
    elif strategy == "Covered Call":
        return [("Stock", 0, 1), ("Call", 1.0 + width_upper, -1)] 
    elif strategy == "Bull Call Spread":
        return [("Call", 1.0, 1 * pos_mult), ("Call", 1.0 + width_upper, -1 * pos_mult)]
    elif strategy == "Call Ratio Backspread":
        return [("Call", 1.0, -1 * pos_mult), ("Call", 1.0 + width_upper, 2 * pos_mult)]
        
    # Stratégies utilisant la borne BASSE (Put OTM)
    elif strategy == "Protective Put":
        return [("Stock", 0, 1), ("Put", 1.0 - width_lower, 1)] 
    elif strategy == "Bear Put Spread":
        return [("Put", 1.0, 1 * pos_mult), ("Put", 1.0 - width_lower, -1 * pos_mult)]
        
    # Stratégies utilisant les DEUX bornes
    elif strategy == "Strangle":
        return [("Call", 1.0 + width_upper, 1 * pos_mult), ("Put", 1.0 - width_lower, 1 * pos_mult)]
    elif strategy == "Butterfly":
        return [("Call", 1.0 - width_lower, 1*pos_mult), ("Call", 1.0, -2*pos_mult), ("Call", 1.0 + width_upper, 1*pos_mult)]
    
    return []

# --- 3. INTERFACE ---

st.title("Derivatives Structuring Tool")

col_params, col_viz = st.columns([1, 3])

with col_params:
    with st.container(border=True):
        st.header("1. Structuration")
        strat_list = ["Call", "Put", "Covered Call", "Protective Put", 
                      "Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", 
                      "Butterfly", "Call Ratio Backspread"]
        
        selected_strat = st.selectbox("Produit", strat_list, index=9)
        
        if selected_strat in ["Covered Call", "Protective Put"]:
            position = "Long"
            st.info("Structure Long Only")
        else:
            position = st.pills("Direction", ["Long", "Short"], default="Long")
            
        st.divider()
        st.header("2. Paramètres Avancés")
        
        # --- LOGIQUE D'AFFICHAGE DYNAMIQUE DES SLIDERS ---
        # On ne montre que les sliders utiles pour la stratégie sélectionnée
        
        # Upper Spread utile pour : Call OTM (Covered Call, Bull Spread, Backspread, Strangle, Butterfly)
        show_upper = selected_strat in ["Covered Call", "Bull Call Spread", "Call Ratio Backspread", "Strangle", "Butterfly"]
        
        # Lower Spread utile pour : Put OTM (Protective Put, Bear Spread, Strangle, Butterfly)
        show_lower = selected_strat in ["Protective Put", "Bear Put Spread", "Strangle", "Butterfly"]

        width_lower = 0.10 # Valeur par défaut si caché
        width_upper = 0.10 # Valeur par défaut si caché

        if not show_upper and not show_lower:
            st.caption("Aucun paramètre d'écart pour cette stratégie (Structure ATM).")
        else:
            col_width1, col_width2 = st.columns(2)
            with col_width1:
                if show_lower:
                    width_lower = st.slider("Lower Spread (-%)", min_value=0.01, max_value=0.50, value=0.10, step=0.01, format="%.2f")
            with col_width2:
                if show_upper:
                    width_upper = st.slider("Upper Spread (+%)", min_value=0.01, max_value=0.50, value=0.15, step=0.01, format="%.2f")

        st.divider()
        st.header("3. Market Data")
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike Central (K)", value=100.0)
        T = st.slider("Maturity (Years)", 0.01, 2.0, 0.5, step=0.01)
        sigma = st.slider("Implied Volatility (sigma)", 0.05, 2.00, 0.35)
        r = st.number_input("Risk Free Rate (r)", value=0.04)

# Calculs
legs_config = get_strategy_legs(selected_strat, K, width_lower, width_upper, position)
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
    with st.expander("Fiche Produit & Analyse", expanded=True):
        st.subheader(f"{selected_strat} ({position})")
        st.text(get_strategy_description(selected_strat, position))

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    cost_label = "Premium (Debit)" if total_price > 0 else "Credit Received"
    kpi1.metric(cost_label, f"{abs(total_price):.2f} $", delta="-Paid" if total_price > 0 else "+Received", delta_color="inverse")
    kpi2.metric("Net Delta", f"{total_delta:.2f}")
    kpi3.metric("Net Gamma", f"{total_gamma:.3f}")
    kpi4.metric("Net Theta", f"{total_theta:.3f}")
    kpi5.metric("Net Vega", f"{total_vega:.2f}")

    st.subheader("Simulateur P&L")
    
    S_range = np.linspace(S * 0.5, S * 1.8, 300)
    
    # Le PnL est : Valeur du portefeuille à maturité - Coût initial
    # Coût initial (total_price) : Si positif on a payé, si négatif on a reçu.
    # Donc PnL = Valeur_Finale - total_price
    pnl_maturity = np.zeros_like(S_range) - total_price 
    
    for leg_type, leg_k, qty in real_legs_details:
        if leg_type == "Call":
            pnl_maturity += np.maximum(S_range - leg_k, 0) * qty
        elif leg_type == "Put":
            pnl_maturity += np.maximum(leg_k - S_range, 0) * qty
        elif leg_type == "Stock":
            pnl_maturity += S_range * qty

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity >= 0), color='#2E8B57', alpha=0.3, interpolate=True)
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity < 0), color='#CD5C5C', alpha=0.3, interpolate=True)
    ax.plot(S_range, pnl_maturity, color="white", linewidth=2.5)
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(S, color='#FFD700', linestyle='--', label=f"Spot: {S}")
    
    for t, k, q in real_legs_details:
        if k > 0:
            ax.axvline(k, color='gray', linestyle=':', alpha=0.5)
            ax.text(k, ax.get_ylim()[1]*0.9, f"{'L' if q>0 else 'S'} {k:.0f}", color='white', ha='center', fontsize=8)

    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='#444444', linestyle=':', linewidth=0.5)
    ax.legend(facecolor='#0E1117', labelcolor='white')
    st.pyplot(fig)

    st.caption("Détail de la structuration")
    legs_data = [{"Type": t, "Strike": f"{k:.2f}" if k > 0 else "Mkt", "Qté": q, "Side": "Long" if q > 0 else "Short"} for t, k, q in real_legs_details]
    st.dataframe(legs_data, use_container_width=True)

st.write("---")
st.markdown("Coded by [Karim MAOUI](https://github.com/KarimMaoui)")

import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pro Option Pricer", layout="wide")

# --- 1. MOTEUR MATHÉMATIQUE (Inchangé) ---

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
        **Structure Produit :** {structure}
        
        **Thèse d'investissement :** {these}
        
        **Contexte Marché :** {contexte}
        """

    desc = {
        "Call": {
            "Long": format_desc("Directionnel Haussier.", "Levier pur.", "Momentum."),
            "Short": format_desc("Vente à découvert.", "Encaissement prime.", "Baissier/Neutre.")
        },
        "Put": {
            "Long": format_desc("Directionnel Baissier.", "Protection ou spéculation.", "Correction."),
            "Short": format_desc("Génération de Rendement.", "Volonté d'acheter bas.", "Neutre/Haussier.")
        },
        "Covered Call": {
            "Long": format_desc("Yield Enhancement.", "Monétisation d'actif.", "Range/Léger haussier."),
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": format_desc("Assurance.", "Plafond de perte.", "Incertitude."),
            "Short": "N/A"
        },
        "Straddle": {
            "Long": format_desc("Volatilité Pure (ATM).", "Explosion du prix indifférente de la direction.", "Earnings/CPI."),
            "Short": format_desc("Vente Volatilité.", "Retour au calme.", "Range.")
        },
        "Strangle": {
            "Long": format_desc("Volatilité (OTM).", "Mouvement violent requis, coût faible.", "Black Swan."),
            "Short": format_desc("Vente Volatilité (Large).", "Marge d'erreur.", "Range large.")
        },
        "Bull Call Spread": {
            "Long": format_desc("Haussier Risque Défini.", "Moins cher qu'un Call.", "Hausse modérée."),
            "Short": format_desc("Crédit Baissier.", "Encaissement.", "Baissier/Neutre.")
        },
        "Bear Put Spread": {
            "Long": format_desc("Baissier Risque Défini.", "Moins cher qu'un Put.", "Baisse modérée."),
            "Short": format_desc("Crédit Haussier.", "Encaissement.", "Haussier/Neutre.")
        },
        "Butterfly": {
            "Long": format_desc("Neutre (Target précis).", "Capture max de Theta.", "Calme plat."),
            "Short": format_desc("Volatilité.", "Sortie de zone.", "Breakout.")
        },
        "Call Ratio Backspread": {
            "Long": format_desc(
                "1 Short ATM / 2 Long OTM.",
                "Gain illimité à la hausse. Souvent monté pour un 'Zéro Coût' (la vente paie les achats).",
                "Volatilité extrême haussière (Commo/Tech)."
            ),
            "Short": format_desc("Contrarian.", "Pari risqué.", "Non standard.")
        }
    }
    return desc.get(strategy, {}).get(position, "N/A")

# --- MODIFICATION MAJEURE ICI : Ajout du paramètre 'width' ---
def get_strategy_legs(strategy, K, width, position="Long"):
    pos_mult = 1 if position == "Long" else -1
    
    # width est un pourcentage (ex: 0.05 pour 5%)
    
    if strategy == "Call":
        return [("Call", 1.0, 1 * pos_mult)]
    elif strategy == "Put":
        return [("Put", 1.0, 1 * pos_mult)]
        
    elif strategy == "Covered Call":
        # On vend le Call à K * (1 + width)
        return [("Stock", 0, 1), ("Call", 1.0 + width, -1)] 
        
    elif strategy == "Protective Put":
        # On achète le Put à K * (1 - width)
        return [("Stock", 0, 1), ("Put", 1.0 - width, 1)] 
        
    elif strategy == "Straddle":
        # Straddle est toujours ATM, le width ne change rien
        return [("Call", 1.0, 1 * pos_mult), ("Put", 1.0, 1 * pos_mult)]
        
    elif strategy == "Strangle":
        # Call OTM (+ width), Put OTM (- width)
        return [("Call", 1.0 + width, 1 * pos_mult), ("Put", 1.0 - width, 1 * pos_mult)]
        
    elif strategy == "Bull Call Spread":
        # Achat ATM, Vente OTM (+ width)
        return [("Call", 1.0, 1 * pos_mult), ("Call", 1.0 + width, -1 * pos_mult)]
        
    elif strategy == "Bear Put Spread":
        # Achat ATM, Vente OTM (- width)
        return [("Put", 1.0, 1 * pos_mult), ("Put", 1.0 - width, -1 * pos_mult)]
        
    elif strategy == "Butterfly":
        # Ailes écartées de 'width'
        return [("Call", 1.0 - width, 1*pos_mult), ("Call", 1.0, -2*pos_mult), ("Call", 1.0 + width, 1*pos_mult)]
        
    elif strategy == "Call Ratio Backspread":
        # C'est ici que tu peux jouer ! 
        # Short ATM (1.0), Long 2x OTM (1.0 + width)
        # Plus 'width' est grand, moins la stratégie coûte cher, mais plus le creux est large.
        return [("Call", 1.0, -1 * pos_mult), ("Call", 1.0 + width, 2 * pos_mult)]
    
    return []

# --- 3. INTERFACE ---

st.title("🛡️ Derivatives Structuring Tool")

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
        
        # --- NOUVEAU SLIDER ---
        # Permet de régler l'écartement des strikes
        width_pct = st.slider("Écart des Strikes (Spread Width)", min_value=0.01, max_value=0.40, value=0.10, step=0.01, format="%.2f")
        st.caption(f"Impact : Les jambes OTM seront à +/- {width_pct*100:.0f}% du strike central.")

        st.divider()
        st.header("3. Market Data")
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike Central (K)", value=100.0)
        T = st.slider("Maturity (Years)", 0.01, 2.0, 0.5, step=0.01)
        sigma = st.slider("Implied Volatility (σ)", 0.05, 2.00, 0.35)
        r = st.number_input("Risk Free Rate (r)", value=0.04)

# Calculs avec le nouveau paramètre 'width_pct'
legs_config = get_strategy_legs(selected_strat, K, width_pct, position)
total_price, total_delta, total_gamma, total_theta, total_vega = 0, 0, 0, 0, 0
real_legs_details = []

for leg_type, strike_mult, qty in legs_config:
    # strike_mult est maintenant le strike exact calculé dans la fonction
    # Mais attention, dans ma fonction j'ai renvoyé des multiplicateurs (ex: 1.0, 1.1)
    # ou j'ai renvoyé directement des formules.
    # Pour garder la logique propre, get_strategy_legs renvoie des MULTIPLICATEURS (1.0 + width).
    
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
    with st.expander("📋 Fiche Produit & Analyse", expanded=True):
        st.subheader(f"{selected_strat} ({position})")
        st.markdown(get_strategy_description(selected_strat, position))

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    cost_label = "Premium (Debit)" if total_price > 0 else "Credit Received"
    kpi1.metric(cost_label, f"{abs(total_price):.2f} $", delta="-Payé" if total_price > 0 else "+Reçu", delta_color="inverse")
    kpi2.metric("Net Delta", f"{total_delta:.2f}")
    kpi3.metric("Net Gamma", f"{total_gamma:.3f}")
    kpi4.metric("Net Theta", f"{total_theta:.3f}")
    kpi5.metric("Net Vega", f"{total_vega:.2f}")

    st.subheader("Simulateur P&L")
    
    S_range = np.linspace(S * 0.5, S * 1.8, 300)
    pnl_maturity = np.zeros_like(S_range) - total_price 
    
    for leg_type, leg_k, qty in real_legs_details:
        if leg_type == "Call":
            pnl_maturity += np.maximum(S_range - leg_k, 0) * qty
        elif leg_type == "Put":
            pnl_maturity += np.maximum(leg_k - S_range, 0) * qty
        elif leg_type == "Stock":
            pnl_maturity += (S_range - S) * qty

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity >= 0), color='#2E8B57', alpha=0.3, interpolate=True)
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity < 0), color='#CD5C5C', alpha=0.3, interpolate=True)
    ax.plot(S_range, pnl_maturity, color="white", linewidth=2.5)
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(S, color='#FFD700', linestyle='--', label=f"Spot: {S}")
    
    # Indicateurs visuels des strikes
    for t, k, q in real_legs_details:
        if k > 0:
            ax.axvline(k, color='gray', linestyle=':', alpha=0.5)
            # Petit texte pour indiquer le strike
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

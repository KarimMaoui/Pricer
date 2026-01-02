import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Derivatives Pricer", layout="wide")

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

# --- 2. LOGIQUE DES STRATÉGIES (VERSION PRO) ---

def get_strategy_description(strategy, position):
    def format_desc(structure, these, contexte):
        return f"""
        Structure Produit :
        {structure}
        
        Thèse d'investissement :
        {these}
        
        Contexte Marché :
        {contexte}
        """

    desc = {
        "Call": {
            "Long": format_desc(
                "Achat Call (Strike K).",
                "Exposition directionnelle haussière avec effet de levier (Gearing). La perte est strictement limitée à la prime payée, tandis que le profit est théoriquement illimité au-delà du point mort (Strike + Prime). Sensibilité positive à la volatilité (Vega Long).",
                "Anticipation d'un mouvement haussier violent et rapide (Momentum). Conviction forte sur un catalyseur à court terme (M&A, Résultats)."
            ),
            "Short": format_desc(
                "Vente Call (Naked).",
                "Stratégie de rendement pur. Le gain est plafonné à la prime reçue. Le risque est illimité si le marché monte. Profil de risque asymétrique défavorable (Negative Skewness).",
                "Marché baissier ou conviction que le niveau de résistance technique ne sera pas franchi. Volatilité implicite jugée excessivement chère."
            )
        },
        "Put": {
            "Long": format_desc(
                "Achat Put (Strike K).",
                "Exposition directionnelle baissière. Utilisé soit pour la spéculation, soit comme couverture (Hedge) d'un portefeuille existant. Fixe un prix de vente plancher.",
                "Anticipation d'une correction de marché ou couverture contre un 'Tail Risk' (Risque de queue de distribution)."
            ),
            "Short": format_desc(
                "Vente Put (Naked).",
                "Stratégie d'accumulation (Target Buying). L'investisseur s'engage à acheter le sous-jacent au Strike K. Il encaisse la prime pour patienter.",
                "Marché neutre à légèrement haussier. Volonté d'entrer sur le titre à un prix décoté par rapport au Spot actuel."
            )
        },
        "Covered Call": {
            "Long": format_desc(
                "Long Sous-jacent + Short Call (OTM).",
                "Stratégie de 'Yield Enhancement' (Amélioration du rendement). La prime reçue agit comme un dividende synthétique et offre un coussin (Buffer) contre une baisse modérée. En contrepartie, l'investisseur renonce à tout gain au-delà du Strike du Call.",
                "Marché neutre ou haussier lent. Idéal quand la volatilité est élevée pour monétiser la détention du titre."
            ),
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": format_desc(
                "Long Sous-jacent + Long Put (OTM/ATM).",
                "Stratégie d'assurance (Synthetic Call). L'investisseur conserve 100% du potentiel de hausse de l'action tout en fixant une perte maximale absolue (Strike du Put - Prime).",
                "Incertitude à court terme (Earnings, Élections) sur une position stratégique long terme qu'on ne souhaite pas solder."
            ),
            "Short": "N/A"
        },
        "Straddle": {
            "Long": format_desc(
                "Achat Call (ATM) + Achat Put (ATM).",
                "Stratégie de Volatilité Pure (Delta Neutre). Pari sur une expansion de la volatilité réalisée supérieure à la volatilité implicite payée. La direction du mouvement importe peu, seule l'amplitude compte.",
                "Événements binaires majeurs (Earnings Surprise, Décision Banques Centrales, CPI) susceptibles de provoquer un gap de prix."
            ),
            "Short": format_desc(
                "Vente Call (ATM) + Vente Put (ATM).",
                "Vente de Volatilité (Short Vega). Pari sur la compression de volatilité et l'érosion du temps (Theta Decay). Risque de perte illimité des deux côtés.",
                "Marché en range strict. Retour au calme attendu après une sur-réaction du marché (Mean Reversion)."
            )
        },
        "Strangle": {
            "Long": format_desc(
                "Achat Call (OTM) + Achat Put (OTM).",
                "Volatilité à moindre coût. Contrairement au Straddle, le point mort est plus éloigné. Nécessite un mouvement de marché très violent pour devenir profitable.",
                "Pari sur un 'Cygne Noir' (Black Swan) ou une rupture de tendance majeure à faible coût initial."
            ),
            "Short": format_desc(
                "Vente Call (OTM) + Vente Put (OTM).",
                "Vente de Volatilité avec marge de sécurité. Profitable tant que le cours reste dans le tunnel formé par les deux strikes. Probabilité de gain élevée, mais risque de queue de distribution (Tail Risk).",
                "Marché latéral (Sideways). Volatilité implicite élevée permettant d'éloigner les bornes tout en gardant une prime attractive."
            )
        },
        "Bull Call Spread": {
            "Long": format_desc(
                "Achat Call (K1) + Vente Call (K2). (K1 < K2).",
                "Optimisation directionnelle. La vente du Call K2 finance une partie de l'achat de K1. Réduit le coût de revient et le point mort, en échange d'un profit capé à K2.",
                "Hausse modérée anticipée. L'investisseur ne croit pas à une explosion au-delà de K2 et souhaite réduire son exposition à une baisse de volatilité."
            ),
            "Short": format_desc(
                "Vente Call (K1) + Achat Call (K2). (Crédit Spread).",
                "Stratégie de Crédit. L'investisseur parie que le marché ne dépassera pas K1. Le gain est limité au crédit reçu initialement.",
                "Marché baissier, neutre, ou butée sur une résistance technique majeure."
            )
        },
        "Bear Put Spread": {
            "Long": format_desc(
                "Achat Put (K1) + Vente Put (K2). (K1 > K2).",
                "Protection ou spéculation baissière à coût réduit. La vente du Put K2 finance l'achat du Put K1. Le profit maximal est atteint si le cours touche K2.",
                "Baisse modérée anticipée (Target précis). Pas de scénario de crash systémique (auquel cas un Put sec serait préférable)."
            ),
            "Short": format_desc(
                "Vente Put (K1) + Achat Put (K2). (Crédit Spread).",
                "Stratégie de Crédit (Bull Put Spread). L'investisseur parie que le marché ne passera pas sous K1. Le risque est défini par l'écart entre les strikes.",
                "Marché haussier, neutre, ou rebond sur un support technique."
            )
        },
        "Butterfly": {
            "Long": format_desc(
                "Achat Call (K1) + Vente 2 Calls (K2) + Achat Call (K3).",
                "Arbitrage de Volatilité (Short Gamma). Stratégie visant à capturer la valeur temps maximale. Le profit est optimal si le cours expire exactement sur le strike central K2.",
                "Marché sans tendance, anémique. Anticipation d'une baisse de volatilité (Volatility Crush)."
            ),
            "Short": format_desc(
                "Vente Call (K1) + Achat 2 Calls (K2) + Vente Call (K3).",
                "Long Volatilité à risque défini. Profitable si le cours sort violemment de la zone centrale, peu importe la direction.",
                "Sortie de congestion imminente (Breakout) attendue."
            )
        },
        "Call Ratio Backspread": {
            "Long": format_desc(
                "Vente 1 Call (ATM) + Achat 2 Calls (OTM).",
                "Stratégie convexe (Long Volatilité). La vente du Call ATM finance l'achat des Calls OTM (souvent à coût nul ou crédit). Profit illimité à la hausse, risque limité si le marché baisse.",
                "**Spécial Commodities / Tech.** Anticipation d'un 'Spike' violent à la hausse (Guerre, Pénurie). On accepte une perte légère si le marché monte doucement (Trap Zone), pour viser le Home Run."
            ),
            "Short": format_desc(
                "Achat 1 Call (ATM) + Vente 2 Calls (OTM).",
                "Stratégie Contrarian très risquée (Naked Short). Pari que la hausse va s'essouffler précisément sur le strike vendu.",
                "Non standard. Déconseillé sans couverture dynamique."
            )
        }
    }
    return desc.get(strategy, {}).get(position, "N/A")

def get_strategy_legs(strategy, K, width_lower, width_upper, position="Long"):
    pos_mult = 1 if position == "Long" else -1
    
    # Stratégies simples (Call/Put/Straddle)
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

st.title("Derivatives Pricer")

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
        
        # LOGIQUE D'AFFICHAGE DYNAMIQUE DES SLIDERS
        show_upper = selected_strat in ["Covered Call", "Bull Call Spread", "Call Ratio Backspread", "Strangle", "Butterfly"]
        show_lower = selected_strat in ["Protective Put", "Bear Put Spread", "Strangle", "Butterfly"]

        width_lower = 0.10
        width_upper = 0.10

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
        # Affichage du texte en format pré-calculé (text simple pour éviter interprétation markdown complexe inutile)
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

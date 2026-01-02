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

# --- 2. LOGIQUE DES STRATÉGIES (VERSION SALLE DE MARCHÉ) ---

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
                "Levier directionnel (Gearing). Perte max limitée à la prime. Profit illimité.",
                "Momentum haussier fort ou anticipation de catalyseur."
            ),
            "Short": format_desc(
                "Vente Call (Naked).",
                "Yield (Short Vega). Gain capé à la prime. Risque illimité.",
                "Marché baissier ou résistance technique."
            )
        },
        "Put": {
            "Long": format_desc(
                "Achat Put (Strike K).",
                "Protection (Floor) ou spéculation baissière.",
                "Correction de marché ou couverture de portefeuille."
            ),
            "Short": format_desc(
                "Vente Put (Naked).",
                "Target Buying. Engagement d'achat du sous-jacent au Strike. Encaissement de prime.",
                "Marché neutre/haussier. Volonté d'entrée à bon compte."
            )
        },
        "Covered Call": {
            "Long": format_desc(
                "Long Stock + Short Call (OTM).",
                "Yield Enhancement. La prime amortit la baisse. Upside capé.",
                "Marché neutre/haussier lent. Volatilité élevée."
            ),
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": format_desc(
                "Long Stock + Long Put (OTM).",
                "Assurance Portefeuille. Perte max connue et fixée.",
                "Incertitude court terme sur position stratégique."
            ),
            "Short": "N/A"
        },
        "Collar": {
            "Long": format_desc(
                "Long Stock + Long Put (Bas) + Short Call (Haut).",
                "Protection financée (Zero Cost). La vente du Call finance l'achat du Put. Le P&L est encadré (Floor/Cap).",
                "Prudence. Volonté de se couvrir sans sortir de cash (Costless Collar)."
            ),
            "Short": "N/A"
        },
        "Risk Reversal": {
            "Long": format_desc(
                "Long Call (Haut) + Short Put (Bas).",
                "Synthétique directionnel financé. On joue le Skew de vol. Réplique le sous-jacent avec moins de capital.",
                "Retournement haussier (Reversal) ou arbitrage de Skew."
            ),
            "Short": format_desc(
                "Short Call (Haut) + Long Put (Bas).",
                "Synthétique baissier financé.",
                "Retournement baissier."
            )
        },
        "Straddle": {
            "Long": format_desc(
                "Achat Call (ATM) + Achat Put (ATM).",
                "Volatilité Pure (Delta Neutre). Pari sur l'amplitude du mouvement.",
                "Earnings, CPI, Banques Centrales."
            ),
            "Short": format_desc(
                "Vente Call (ATM) + Vente Put (ATM).",
                "Short Volatilité. Pari sur le retour au calme.",
                "Range strict. Post-événement."
            )
        },
        "Strangle": {
            "Long": format_desc(
                "Achat Call (OTM) + Achat Put (OTM).",
                "Volatilité Low-Cost. Nécessite un mouvement violent.",
                "Black Swan / Rupture."
            ),
            "Short": format_desc(
                "Vente Call (OTM) + Vente Put (OTM).",
                "Short Volatilité Large. Marge d'erreur plus grande que le Straddle.",
                "Marché latéral (Sideways)."
            )
        },
        "Strap": {
            "Long": format_desc(
                "Achat 2 Calls (ATM) + Achat 1 Put (ATM).",
                "Volatilité avec Biais Haussier. Profitable si le marché explose, mais gain doublé à la hausse (Delta positif).",
                "Volatilité attendue mais conviction haussière dominante."
            ),
            "Short": format_desc(
                "Vente 2 Calls (ATM) + Vente 1 Put (ATM).",
                "Short Volatilité Biais Baissier. Très risqué à la hausse.",
                "Marché calme ou baissier lent."
            )
        },
        "Condor": {
            "Long": format_desc(
                "Achat Call (K1) + Vente Call (K2) + Vente Call (K3) + Achat Call (K4).",
                "Arbitrage de Volatilité (Zone de profit large). Capture de Theta optimale si le cours reste entre K2 et K3.",
                "Marché très calme (Indice range trading)."
            ),
            "Short": format_desc(
                "Vente K1 + Achat K2 + Achat K3 + Vente K4.",
                "Volatilité (Breakout). Profitable si le cours sort de la zone centrale.",
                "Sortie de range attendue."
            )
        },
        "Bull Call Spread": {
            "Long": format_desc(
                "Achat Call (K1) + Vente Call (K2).",
                "Directionnel optimisé. Coût réduit, profit capé.",
                "Hausse modérée."
            ),
            "Short": format_desc(
                "Vente Call (K1) + Achat Call (K2).",
                "Crédit Spread (Baissier).",
                "Résistance technique."
            )
        },
        "Bear Put Spread": {
            "Long": format_desc(
                "Achat Put (K1) + Vente Put (K2).",
                "Directionnel optimisé. Coût réduit, profit capé.",
                "Baisse modérée."
            ),
            "Short": format_desc(
                "Vente Put (K1) + Achat Put (K2).",
                "Crédit Spread (Haussier).",
                "Support technique."
            )
        },
        "Seagull": {
            "Long": format_desc(
                "Bull Call Spread + Vente Put (Bas).",
                "Financement total de la hausse. Souvent Zero Premium. Risque baissier présent.",
                "Marché haussier, prêt à acheter au strike bas."
            ),
            "Short": "N/A"
        },
        "Butterfly": {
            "Long": format_desc(
                "Achat K1 + Vente 2x K2 + Achat K3.",
                "Arbitrage Volatilité (Short Gamma). Profit max sur un point précis (K2).",
                "Marché anémique."
            ),
            "Short": "N/A"
        },
        "Call Ratio Backspread": {
            "Long": format_desc(
                "Vente 1 Call (ATM) + Achat 2 Calls (OTM).",
                "Volatilité convexe haussière. Gain illimité.",
                "Explosion haussière type Commodities."
            ),
            "Short": "N/A"
        },
        "Put Ratio Backspread": {
            "Long": format_desc(
                "Vente 1 Put (ATM) + Achat 2 Puts (OTM).",
                "Protection Crash aggressive. Gain massif sur krach.",
                "Couverture Krach."
            ),
            "Short": "N/A"
        },
        "Synthetic Long": {
            "Long": format_desc(
                "Achat Call (ATM) + Vente Put (ATM).",
                "Réplication Delta One. Comportement identique à l'action.",
                "Exposition linéaire sans cash."
            ),
            "Short": format_desc(
                "Vente Call (ATM) + Achat Put (ATM).",
                "Short synthétique.",
                "Baisse anticipée."
            )
        }
    }
    return desc.get(strategy, {}).get(position, "N/A")

def get_strategy_legs(strategy, K, width_lower, width_upper, position="Long"):
    pos_mult = 1 if position == "Long" else -1
    
    # 1. Stratégies Simples
    if strategy == "Call":
        return [("Call", 1.0, 1 * pos_mult)]
    elif strategy == "Put":
        return [("Put", 1.0, 1 * pos_mult)]
    elif strategy == "Straddle":
        return [("Call", 1.0, 1 * pos_mult), ("Put", 1.0, 1 * pos_mult)]
    elif strategy == "Synthetic Long":
        return [("Call", 1.0, 1 * pos_mult), ("Put", 1.0, -1 * pos_mult)]
    
    # 2. NOUVEAU : STRAP (2 Calls / 1 Put)
    elif strategy == "Strap":
        # 2 Calls ATM, 1 Put ATM
        return [("Call", 1.0, 2 * pos_mult), ("Put", 1.0, 1 * pos_mult)]

    # 3. Stratégies bornées HAUTE (Call OTM)
    elif strategy == "Covered Call":
        return [("Stock", 0, 1), ("Call", 1.0 + width_upper, -1)] 
    elif strategy == "Bull Call Spread":
        return [("Call", 1.0, 1 * pos_mult), ("Call", 1.0 + width_upper, -1 * pos_mult)]
    elif strategy == "Call Ratio Backspread":
        return [("Call", 1.0, -1 * pos_mult), ("Call", 1.0 + width_upper, 2 * pos_mult)]
        
    # 4. Stratégies bornées BASSE (Put OTM)
    elif strategy == "Protective Put":
        return [("Stock", 0, 1), ("Put", 1.0 - width_lower, 1)] 
    elif strategy == "Bear Put Spread":
        return [("Put", 1.0, 1 * pos_mult), ("Put", 1.0 - width_lower, -1 * pos_mult)]
    elif strategy == "Put Ratio Backspread":
        return [("Put", 1.0, -1 * pos_mult), ("Put", 1.0 - width_lower, 2 * pos_mult)]
        
    # 5. Stratégies DOUBLE bornes (Haut et Bas)
    elif strategy == "Strangle":
        return [("Call", 1.0 + width_upper, 1 * pos_mult), ("Put", 1.0 - width_lower, 1 * pos_mult)]
    elif strategy == "Butterfly":
        return [("Call", 1.0 - width_lower, 1*pos_mult), ("Call", 1.0, -2*pos_mult), ("Call", 1.0 + width_upper, 1*pos_mult)]
    
    # 6. NOUVEAU : CONDOR (Corps large)
    elif strategy == "Condor":
        # On utilise width_lower pour l'aile gauche et width_upper pour l'aile droite
        # On décale le corps de 2% pour créer le "plat" du Condor
        body_gap = 0.02
        return [
            ("Call", 1.0 - width_lower - body_gap, 1*pos_mult), # Aile Gauche
            ("Call", 1.0 - body_gap, -1*pos_mult), # Corps Gauche
            ("Call", 1.0 + body_gap, -1*pos_mult), # Corps Droit
            ("Call", 1.0 + width_upper + body_gap, 1*pos_mult) # Aile Droite
        ]
    
    elif strategy == "Collar":
        return [("Stock", 0, 1), ("Put", 1.0 - width_lower, 1), ("Call", 1.0 + width_upper, -1)]
        
    elif strategy == "Risk Reversal":
        return [("Call", 1.0 + width_upper, 1 * pos_mult), ("Put", 1.0 - width_lower, -1 * pos_mult)]
        
    elif strategy == "Seagull":
        return [("Call", 1.0, 1), ("Call", 1.0 + width_upper, -1), ("Put", 1.0 - width_lower, -1)]

    return []

# --- 3. INTERFACE ---

st.title("Derivatives Pricer")

col_params, col_viz = st.columns([1, 3])

with col_params:
    with st.container(border=True):
        st.header("1. Structuration")
        strat_list = [
            "Call", "Put", "Synthetic Long",
            "Covered Call", "Protective Put", "Collar",
            "Straddle", "Strangle", "Strap", "Butterfly", "Condor",
            "Bull Call Spread", "Bear Put Spread", "Seagull", "Risk Reversal",
            "Call Ratio Backspread", "Put Ratio Backspread"
        ]
        
        selected_strat = st.selectbox("Produit", strat_list, index=0)
        
        if selected_strat in ["Covered Call", "Protective Put", "Collar", "Seagull"]:
            position = "Long"
            st.info("Structure Standard (Long)")
        else:
            position = st.pills("Direction", ["Long", "Short"], default="Long")
            
        st.divider()
        st.header("2. Paramètres Avancés")
        
        # LOGIQUE D'AFFICHAGE DYNAMIQUE DES SLIDERS
        
        upper_strategies = ["Covered Call", "Bull Call Spread", "Call Ratio Backspread", 
                            "Strangle", "Butterfly", "Condor", "Collar", "Risk Reversal", "Seagull"]
        show_upper = selected_strat in upper_strategies
        
        lower_strategies = ["Protective Put", "Bear Put Spread", "Put Ratio Backspread", 
                            "Strangle", "Butterfly", "Condor", "Collar", "Risk Reversal", "Seagull"]
        show_lower = selected_strat in lower_strategies

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
        sigma = st.slider("Implied Volatility (sigma)", 0.05, 2.00, 0.30)
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
            # Affichage propre des strikes avec logique L/S
            ax.text(k, ax.get_ylim()[1]*0.95, f"{k:.0f}", color='white', ha='center', fontsize=7, alpha=0.7)

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

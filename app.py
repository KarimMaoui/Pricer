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

# --- 2. LOGIQUE DES STRATÉGIES (VERSION INSTITUTIONNELLE) ---

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
                "Exposition directionnelle haussière avec effet de levier (Gearing). Perte limitée à la prime, profit illimité. Sensible à la hausse de volatilité.",
                "Anticipation d'un mouvement haussier violent (Momentum). Conviction forte sur un catalyseur."
            ),
            "Short": format_desc(
                "Vente Call (Naked).",
                "Stratégie de rendement (Short Vega). Gain plafonné à la prime. Risque illimité.",
                "Marché baissier ou butée sur résistance technique majeure."
            )
        },
        "Put": {
            "Long": format_desc(
                "Achat Put (Strike K).",
                "Exposition directionnelle baissière. Utilisé pour la spéculation ou la couverture (Floor).",
                "Anticipation d'une correction ou couverture 'Tail Risk'."
            ),
            "Short": format_desc(
                "Vente Put (Naked).",
                "Accumulation (Target Buying). Engagement d'achat du sous-jacent au Strike K. Encaissement de prime.",
                "Marché neutre/haussier. Volonté d'entrée à un prix décoté."
            )
        },
        "Covered Call": {
            "Long": format_desc(
                "Long Sous-jacent + Short Call (OTM).",
                "Yield Enhancement. La prime reçue amortit les baisses légères. Renoncement à l'upside au-delà du strike.",
                "Marché neutre ou haussier lent. Volatilité implicite élevée."
            ),
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": format_desc(
                "Long Sous-jacent + Long Put (OTM/ATM).",
                "Assurance Portefeuille. Conservation du potentiel haussier avec perte maximale garantie.",
                "Incertitude à court terme sur une position stratégique."
            ),
            "Short": "N/A"
        },
        "Collar": {
            "Long": format_desc(
                "Long Sous-jacent + Long Put (Bas) + Short Call (Haut).",
                "Protection à coût réduit (voire Zéro Coût). La vente du Call finance l'achat du Put. Le P&L est encadré (Floor et Cap).",
                "Détention long terme prudente. Volonté de se couvrir contre un krach sans débourser de cash."
            ),
            "Short": "N/A"
        },
        "Risk Reversal": {
            "Long": format_desc(
                "Long Call (Haut) + Short Put (Bas).",
                "Stratégie directionnelle synthétique financée. On joue le Skew de volatilité. Réplique une position longue sur l'action mais avec une marge initiale réduite.",
                "Retournement de tendance haussier (Reversal) ou arbitrage de Skew (Calls pas chers, Puts chers)."
            ),
            "Short": format_desc(
                "Short Call (Haut) + Long Put (Bas).",
                "Pari directionnel baissier financé. On finance la protection (Put) par la vente de potentiel (Call).",
                "Retournement de tendance baissier."
            )
        },
        "Straddle": {
            "Long": format_desc(
                "Achat Call (ATM) + Achat Put (ATM).",
                "Volatilité Pure (Delta Neutre). Pari sur une amplitude de mouvement supérieure à ce que le marché price.",
                "Événements binaires (Earnings, CPI, Banques Centrales)."
            ),
            "Short": format_desc(
                "Vente Call (ATM) + Vente Put (ATM).",
                "Vente de Volatilité. Pari sur le retour au calme (Mean Reversion) et l'érosion du temps.",
                "Marché en range strict. Post-événement."
            )
        },
        "Strangle": {
            "Long": format_desc(
                "Achat Call (OTM) + Achat Put (OTM).",
                "Volatilité Low-Cost. Nécessite un mouvement plus violent que le Straddle pour être profitable.",
                "Scénario 'Black Swan' ou rupture technique majeure."
            ),
            "Short": format_desc(
                "Vente Call (OTM) + Vente Put (OTM).",
                "Vente de Volatilité Large. Profitable tant que le cours reste dans le tunnel.",
                "Marché latéral (Sideways) avec volatilité élevée."
            )
        },
        "Bull Call Spread": {
            "Long": format_desc(
                "Achat Call (K1) + Vente Call (K2).",
                "Optimisation directionnelle haussière. Coût réduit, profit capé.",
                "Hausse modérée anticipée."
            ),
            "Short": format_desc(
                "Vente Call (K1) + Achat Call (K2).",
                "Crédit Spread (Baissier). Encaissement de prime si le marché ne monte pas.",
                "Résistance technique."
            )
        },
        "Bear Put Spread": {
            "Long": format_desc(
                "Achat Put (K1) + Vente Put (K2).",
                "Optimisation directionnelle baissière. Coût réduit, profit capé.",
                "Baisse modérée anticipée."
            ),
            "Short": format_desc(
                "Vente Put (K1) + Achat Put (K2).",
                "Crédit Spread (Haussier). Encaissement de prime si le marché ne baisse pas.",
                "Support technique."
            )
        },
        "Seagull": {
            "Long": format_desc(
                "Bull Call Spread (Achat K1/Vente K2) + Vente Put (K3).",
                "Produit Structuré. Financement total de la hausse (Call Spread) par la vente du Put. Souvent 'Zero Premium'. Risque baissier similaire à l'action.",
                "Marché haussier, mais on est prêt à acheter le titre si le marché s'effondre (au Strike K3)."
            ),
            "Short": "N/A (Structure complexe rarement shortée telle quelle)."
        },
        "Butterfly": {
            "Long": format_desc(
                "Achat K1 + Vente 2x K2 + Achat K3.",
                "Arbitrage de Volatilité (Short Gamma). Capture maximale de Theta sur un point précis.",
                "Marché anémique, baisse de volatilité."
            ),
            "Short": "N/A"
        },
        "Call Ratio Backspread": {
            "Long": format_desc(
                "Vente 1 Call (ATM) + Achat 2 Calls (OTM).",
                "Volatilité convexe haussière. Gain illimité, risque limité (souvent crédit initial).",
                "Explosion haussière type Commodities."
            ),
            "Short": "N/A"
        },
        "Put Ratio Backspread": {
            "Long": format_desc(
                "Vente 1 Put (ATM) + Achat 2 Puts (OTM).",
                "Protection Crash aggressive. Si le marché baisse un peu, on perd. Si le marché s'effondre, gain massif.",
                "Couverture contre un Krach systémique à moindre frais."
            ),
            "Short": "N/A"
        },
        "Synthetic Long": {
            "Long": format_desc(
                "Achat Call (ATM) + Vente Put (ATM).",
                "Réplication synthétique (Delta One). Comportement identique à l'action mais utilisation du capital (margin) très faible.",
                "Volonté d'exposition linéaire sans immobiliser de cash."
            ),
            "Short": format_desc(
                "Vente Call (ATM) + Achat Put (ATM).",
                "Position courte synthétique. Permet de shorter sans emprunt de titres.",
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

    # 2. Stratégies bornées HAUTE (Call OTM)
    elif strategy == "Covered Call":
        return [("Stock", 0, 1), ("Call", 1.0 + width_upper, -1)] 
    elif strategy == "Bull Call Spread":
        return [("Call", 1.0, 1 * pos_mult), ("Call", 1.0 + width_upper, -1 * pos_mult)]
    elif strategy == "Call Ratio Backspread":
        return [("Call", 1.0, -1 * pos_mult), ("Call", 1.0 + width_upper, 2 * pos_mult)]
        
    # 3. Stratégies bornées BASSE (Put OTM)
    elif strategy == "Protective Put":
        return [("Stock", 0, 1), ("Put", 1.0 - width_lower, 1)] 
    elif strategy == "Bear Put Spread":
        return [("Put", 1.0, 1 * pos_mult), ("Put", 1.0 - width_lower, -1 * pos_mult)]
    elif strategy == "Put Ratio Backspread":
        # Vente Put ATM, Achat 2 Puts OTM (Plus bas)
        return [("Put", 1.0, -1 * pos_mult), ("Put", 1.0 - width_lower, 2 * pos_mult)]
        
    # 4. Stratégies DOUBLE bornes (Haut et Bas)
    elif strategy == "Strangle":
        return [("Call", 1.0 + width_upper, 1 * pos_mult), ("Put", 1.0 - width_lower, 1 * pos_mult)]
    elif strategy == "Butterfly":
        return [("Call", 1.0 - width_lower, 1*pos_mult), ("Call", 1.0, -2*pos_mult), ("Call", 1.0 + width_upper, 1*pos_mult)]
    
    # 5. NOUVELLES STRATEGIES COMPLEXES
    elif strategy == "Collar":
        # Long Stock + Long Put (Bas) + Short Call (Haut)
        return [("Stock", 0, 1), ("Put", 1.0 - width_lower, 1), ("Call", 1.0 + width_upper, -1)]
        
    elif strategy == "Risk Reversal":
        # Long Call (Haut), Short Put (Bas) -> Bullish RR
        return [("Call", 1.0 + width_upper, 1 * pos_mult), ("Put", 1.0 - width_lower, -1 * pos_mult)]
        
    elif strategy == "Seagull":
        # Bull Call Spread (Achat ATM, Vente Haut) + Vente Put (Bas)
        # Structure Bullish standard
        return [("Call", 1.0, 1), ("Call", 1.0 + width_upper, -1), ("Put", 1.0 - width_lower, -1)]

    return []

# --- 3. INTERFACE ---

st.title("Derivatives Pricer")

col_params, col_viz = st.columns([1, 3])

with col_params:
    with st.container(border=True):
        st.header("1. Structuration")
        # Liste enrichie
        strat_list = [
            "Call", "Put", "Synthetic Long", # Basique
            "Covered Call", "Protective Put", "Collar", # Action + Option
            "Straddle", "Strangle", "Butterfly", # Volatilité
            "Bull Call Spread", "Bear Put Spread", "Seagull", "Risk Reversal", # Directionnel Structuré
            "Call Ratio Backspread", "Put Ratio Backspread" # Anti-Crash / Convexité
        ]
        
        selected_strat = st.selectbox("Produit", strat_list, index=0)
        
        # Gestion des stratégies "Long Only" (celles qui incluent du Stock ou complexes)
        if selected_strat in ["Covered Call", "Protective Put", "Collar", "Seagull"]:
            position = "Long"
            st.info("Structure Standard (Long)")
        else:
            position = st.pills("Direction", ["Long", "Short"], default="Long")
            
        st.divider()
        st.header("2. Paramètres Avancés")
        
        # LOGIQUE D'AFFICHAGE DYNAMIQUE DES SLIDERS
        
        # Upper Spread utile pour : Call OTM
        upper_strategies = ["Covered Call", "Bull Call Spread", "Call Ratio Backspread", 
                            "Strangle", "Butterfly", "Collar", "Risk Reversal", "Seagull"]
        show_upper = selected_strat in upper_strategies
        
        # Lower Spread utile pour : Put OTM
        lower_strategies = ["Protective Put", "Bear Put Spread", "Put Ratio Backspread", 
                            "Strangle", "Butterfly", "Collar", "Risk Reversal", "Seagull"]
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
    
    # Range élargi pour bien voir les produits exotiques
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
            # Affichage discret des strikes sur le graph
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

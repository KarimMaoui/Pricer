import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Derivatives Pricer", layout="wide")

# --- 1. MOTEUR MATHÉMATIQUE (Inchangé) ---
def black_scholes(S, K, T, r, sigma, q, option_type="Call"):
    if option_type == "Stock": return S 
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def get_greeks(S, K, T, r, sigma, q, option_type="Call"):
    if option_type == "Stock": return 1.0, 0.0, 0.0, 0.0 
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
        Structure Produit :
        {structure}
        
        Thèse d'investissement :
        {these}
        
        Contexte Marché :
        {contexte}
        """
    # (Je garde les descriptions précédentes qui étaient bonnes, focus sur les grecques plus bas)
    desc = {
        "Call": {"Long": format_desc("Achat Call (K).", "Levier directionnel.", "Momentum."), "Short": format_desc("Vente Call.", "Yield.", "Baissier.")},
        "Put": {"Long": format_desc("Achat Put (K).", "Protection/Spéculation.", "Correction."), "Short": format_desc("Vente Put.", "Target Buying.", "Neutre/Haussier.")},
        "Covered Call": {"Long": format_desc("Long Stock + Short Call.", "Yield Enhancement.", "Neutre/Haussier lent."), "Short": "N/A"},
        "Protective Put": {"Long": format_desc("Long Stock + Long Put.", "Assurance.", "Incertitude."), "Short": "N/A"},
        "Collar": {"Long": format_desc("Long Stock + Long Put (Bas) + Short Call (Haut).", "Zero Cost Protection.", "Prudence."), "Short": "N/A"},
        "Risk Reversal": {"Long": format_desc("Long Call + Short Put.", "Synthétique financé.", "Reversal."), "Short": format_desc("Short Call + Long Put.", "Synthétique baissier.", "Reversal.")},
        "Straddle": {"Long": format_desc("Long Call ATM + Long Put ATM.", "Volatilité Pure.", "Event."), "Short": format_desc("Short Call ATM + Short Put ATM.", "Short Vol.", "Range.")},
        "Strangle": {"Long": format_desc("Long Call OTM + Long Put OTM.", "Volatilité Low-Cost.", "Black Swan."), "Short": format_desc("Short Call OTM + Short Put OTM.", "Short Vol Large.", "Sideways.")},
        "Strap": {"Long": format_desc("2 Long Calls + 1 Long Put.", "Volatilité Biais Haussier.", "Volatilité + Hausse."), "Short": format_desc("2 Short Calls + 1 Short Put.", "Short Vol Biais Baissier.", "Calme.")},
        "Condor": {"Long": format_desc("Long Wings + Short Body.", "Arbitrage Volatilité.", "Range."), "Short": format_desc("Short Wings + Long Body.", "Volatilité.", "Breakout.")},
        "Bull Call Spread": {"Long": format_desc("Long Call K1 + Short Call K2.", "Directionnel optimisé.", "Hausse modérée."), "Short": format_desc("Short Call K1 + Long Call K2.", "Crédit Spread.", "Résistance.")},
        "Bear Put Spread": {"Long": format_desc("Long Put K1 + Short Put K2.", "Directionnel optimisé.", "Baisse modérée."), "Short": format_desc("Short Put K1 + Long Put K2.", "Crédit Spread.", "Support.")},
        "Seagull": {"Long": format_desc("Bull Call Spread + Short Put.", "Hausse financée.", "Haussier + Target Buying."), "Short": "N/A"},
        "Butterfly": {"Long": format_desc("Long K1 + Short 2x K2 + Long K3.", "Short Gamma.", "Marché anémique."), "Short": "N/A"},
        "Call Ratio Backspread": {"Long": format_desc("Short 1 Call + Long 2 Calls.", "Volatilité convexe.", "Explosion."), "Short": "N/A"},
        "Put Ratio Backspread": {"Long": format_desc("Short 1 Put + Long 2 Puts.", "Protection Crash.", "Krach."), "Short": "N/A"},
        "Synthetic Long": {"Long": format_desc("Long Call + Short Put.", "Delta One.", "Linéaire."), "Short": format_desc("Short Call + Long Put.", "Short synthétique.", "Baisse.")}
    }
    return desc.get(strategy, {}).get(position, "N/A")

# --- NOUVELLE FONCTION : EXPLICATION TECHNIQUE DES GRECQUES ---
def get_greeks_profile(strategy, position):
    
    profiles = {
        "Call": {
            "Long": ("Positif. Delta = Probabilité approximative d'exercice.", "Positif. Accélération maximale ATM.", "Négatif. L'option est un actif périssable.", "Positif. Vega maximal ATM."),
            "Short": ("Négatif. Vous êtes contre le marché.", "Négatif. Risque de 'Gap' contre vous.", "Positif. Vous encaissez la valeur temps.", "Négatif. La baisse de vol réduit votre coût de rachat.")
        },
        "Put": {
            "Long": ("Négatif. Delta tend vers -1 si ITM.", "Positif. L'option devient plus sensible si le marché baisse.", "Négatif. Coût de portage.", "Positif. Le Put prend de la valeur si la peur monte."),
            "Short": ("Positif. Delta tend vers 0 si OTM.", "Négatif. Risque accéléré à la baisse.", "Positif. Rente quotidienne.", "Négatif. Vous vendez de l'assurance.")
        },
        "Covered Call": {
            "Long": ("Positif réduit. Le Short Call K2 freine le Delta du stock (1.0).", "Négatif. Le Gamma du Short Call domine (le Stock a un Gamma de 0).", "Positif. Seule la jambe Short Call génère du Theta.", "Négatif. Si la Vol monte, le Call vendu devient plus cher à racheter."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Protective Put": {
            "Long": ("Positif. Le Delta du Put (-0.x) réduit le Delta du Stock (1.0).", "Positif. Le Gamma du Put vous rend plus long quand le marché baisse.", "Négatif. Vous payez la prime d'assurance.", "Positif. Votre protection vaut plus cher si la vol monte."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Collar": {
            "Long": ("Positif. Encadré par le Call vendu (Haut) et le Put acheté (Bas).", "Variable. Long Gamma en bas (Put), Short Gamma en haut (Call).", "Variable. Dépend si la prime reçue (Call) couvre la prime payée (Put).", "Négatif (Souvent). On vend souvent un Call plus riche en Vol (Skew) que le Put."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Straddle": {
            "Long": ("Neutre (si ATM). Doit être Delta-Hedgé dynamiquement.", "Positif Fort. Cumul des Gamma du Call et du Put. Max ATM.", "Négatif Fort. Vous payez deux primes temporelles.", "Positif Fort. Pure exposition Volatilité."),
            "Short": ("Neutre.", "Négatif Fort. Danger immédiat si le spot décale.", "Positif Fort. Gain maximal si le spot ne bouge pas.", "Négatif Fort. Short Vol pur.")
        },
        "Strangle": {
            "Long": ("Neutre.", "Positif. Moins fort que le Straddle car les strikes sont OTM (Gamma plus faible).", "Négatif. Moins coûteux que le Straddle.", "Positif. Sensibilité Vega présente mais plus faible qu'ATM."),
            "Short": ("Neutre.", "Négatif.", "Positif.", "Négatif.")
        },
        "Bull Call Spread": {
            "Long": ("Positif. Différence entre le Delta K1 (fort) et K2 (faible).", "Positif/Négatif. Flip de Gamma : Long Gamma en bas, Short Gamma en haut.", "Négatif/Positif. Vous payez du Theta sur K1, vous en recevez sur K2.", "Variable. Long Vega sur K1, Short sur K2. Souvent Net Long Vega."),
            "Short": ("Négatif.", "Négatif/Positif.", "Positif/Négatif.", "Variable.")
        },
        "Bear Put Spread": {
            "Long": ("Négatif.", "Positif (Haut) / Négatif (Bas).", "Négatif (Haut) / Positif (Bas).", "Variable. Long Vega sur K1, Short sur K2."),
            "Short": ("Positif.", "Négatif (Haut) / Positif (Bas).", "Positif (Haut) / Négatif (Bas).", "Variable.")
        },
        "Seagull": {
            "Long": ("Positif. Similaire au sous-jacent + effet levier du Call Spread.", "Négatif (Souvent). Le Short Put et le Short Call (Bornes) écrasent le Gamma du Call acheté.", "Positif. Les deux ventes financent largement le Theta du Call acheté.", "Négatif (Souvent). Vous vendez 2 pattes (Put bas + Call haut). La somme des Vega vendus dépasse le Vega acheté."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Butterfly": {
            "Long": ("Neutre.", "Négatif (Zone profit). La vente de 2 ATM (Gamma Max) domine l'achat des ailes OTM.", "Positif Fort. Le Theta des 2 options vendues finance largement les ailes.", "Négatif. Vous êtes Short Volatilité au cœur de la structure."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Call Ratio Backspread": {
            "Long": ("Variable (Souvent Positif).", "Positif Fort. Vous possédez 2 Calls pour 1 vendu. La convexité est doublée.", "Variable.", "Positif. Quantité : 2x Vega OTM > 1x Vega ATM."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Put Ratio Backspread": {
            "Long": ("Variable (Souvent Négatif).", "Positif Fort. Convexité doublée à la baisse.", "Variable.", "Positif. Quantité : 2x Vega OTM > 1x Vega ATM."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Risk Reversal": {
            "Long": ("Positif. Cumul du Delta Call et Delta Put (Short Put = Delta positif).", "Neutre. Les Gamma sont souvent éloignés et faibles.", "Neutre.", "Neutre/Variable. Arbitrage de Skew : si Put plus cher que Call, on est Short Vega."),
            "Short": ("Négatif.", "Neutre.", "Neutre.", "Neutre.")
        },
        "Synthetic Long": {
            "Long": ("Positif (100%). Delta fixe de 1.0 (Call Delta + Put Delta).", "Neutre. Gamma Call et Gamma Put s'annulent.", "Neutre. Theta Call et Put s'annulent.", "Neutre. Vega Call et Put s'annulent."),
            "Short": ("Négatif (100%).", "Neutre.", "Neutre.", "Neutre.")
        },
        "Condor": {
            "Long": ("Neutre.", "Négatif. Zone centrale Short Gamma (Vente K2/K3).", "Positif. Gain temps maximal sur le plateau.", "Négatif. Short Volatilité sur toute la zone de profit."),
            "Short": ("Neutre.", "Positif. Zone centrale Long Gamma.", "Négatif.", "Positif.")
        },
        "Strap": {
            "Long": ("Positif Fort. 2 Calls vs 1 Put crée un biais haussier net.", "Positif Fort. 3 options achetées = Gamma massif.", "Négatif Fort. 3 primes à payer chaque jour.", "Positif Fort. Exposition Vega triplée."),
            "Short": ("Négatif Fort.", "Négatif Fort.", "Positif Fort.", "Négatif Fort.")
        }
    }
    
    return profiles.get(strategy, {}).get(position, ("N/A", "N/A", "N/A", "N/A"))

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
    elif strategy == "Strap":
        return [("Call", 1.0, 2 * pos_mult), ("Put", 1.0, 1 * pos_mult)]

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
        return [("Put", 1.0, -1 * pos_mult), ("Put", 1.0 - width_lower, 2 * pos_mult)]
        
    # 4. Stratégies DOUBLE bornes (Haut et Bas)
    elif strategy == "Strangle":
        return [("Call", 1.0 + width_upper, 1 * pos_mult), ("Put", 1.0 - width_lower, 1 * pos_mult)]
    elif strategy == "Butterfly":
        return [("Call", 1.0 - width_lower, 1*pos_mult), ("Call", 1.0, -2*pos_mult), ("Call", 1.0 + width_upper, 1*pos_mult)]
    elif strategy == "Condor":
        body_gap = 0.02
        return [("Call", 1.0 - width_lower - body_gap, 1*pos_mult), ("Call", 1.0 - body_gap, -1*pos_mult), ("Call", 1.0 + body_gap, -1*pos_mult), ("Call", 1.0 + width_upper + body_gap, 1*pos_mult)]
    
    # 5. Stratégies Complexes
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
        T = st.slider("Maturity (Years)", 0.01, 5.0, 1.0, step=0.01)
        sigma = st.slider("Implied Volatility (sigma)", 0.05, 5.00, 0.30)
        r = st.number_input("Risk Free Rate (r)", value=0.02)

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

    # --- AFFICHAGE PRO DES GRECQUES ---
    st.divider()
    st.subheader("Analyse des Risques (Sensibilités)")
    
    txt_delta, txt_gamma, txt_theta, txt_vega = get_greeks_profile(selected_strat, position)
    
    risk1, risk2, risk3, risk4 = st.columns(4)
    
    with risk1:
        st.markdown("**Delta (Direction)**")
        st.info(txt_delta)
    with risk2:
        st.markdown("**Gamma (Convexité)**")
        st.info(txt_gamma)
    with risk3:
        st.markdown("**Theta (Temps)**")
        st.info(txt_theta)
    with risk4:
        st.markdown("**Vega (Volatilité)**")
        st.info(txt_vega)

st.write("---")
st.markdown("Coded by [Karim MAOUI](https://github.com/KarimMaoui)")

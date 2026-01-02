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

# --- 2. LOGIQUE DES STRATÉGIES (Descriptions Pros) ---

def get_strategy_description(strategy, position):
    # Formatage standardisé pour l'affichage
    def format_desc(structure, these, contexte):
        return f"""
        **Structure Produit :** {structure}
        
        **Thèse d'investissement :** {these}
        
        **Contexte Marché :** {contexte}
        """

    desc = {
        "Call": {
            "Long": format_desc(
                "Produit Directionnel Haussier (Long Delta, Long Vega).",
                "Exposition à la hausse avec effet de levier. Le risque est limité à la prime payée, le profit est théoriquement illimité. Sensible positivement à une hausse de la volatilité.",
                "Anticipation d'un mouvement haussier rapide (momentum) ou d'un catalyseur à court terme."
            ),
            "Short": format_desc(
                "Vente à découvert (Short Delta, Short Vega).",
                "Encaissement de prime pur. Profit maximum limité à la prime reçue. Risque de perte illimité si le marché explose à la hausse.",
                "Conviction baissière forte ou marché dont la volatilité est jugée excessivement chère."
            )
        },
        "Put": {
            "Long": format_desc(
                "Produit Directionnel Baissier / Protection (Short Delta, Long Vega).",
                "Profit généré par la baisse du sous-jacent. Agit comme une assurance (plancher) ou un outil de spéculation baissière.",
                "Correction de marché anticipée, couverture de portefeuille actions, ou 'Tail Risk Hedging'."
            ),
            "Short": format_desc(
                "Génération de Rendement (Long Delta, Short Vega).",
                "Encaissement de prime avec obligation d'acheter le sous-jacent au strike si exercé. Profil de risque similaire à la détention d'actions, mais avec un upside capé.",
                "Marché neutre à légèrement haussier. Volonté d'acquérir le sous-jacent à un prix décoté (Target Buying)."
            )
        },
        "Covered Call": {
            "Long": format_desc(
                "Yield Enhancement (Long Stock + Short Call OTM).",
                "Monétisation de la détention d'actifs. On renonce au potentiel de hausse au-delà du strike en échange d'un revenu immédiat (prime) qui amortit les petites baisses.",
                "Marché en range ou légèrement haussier. Volatilité implicite élevée permettant de vendre des Calls chers."
            ),
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": format_desc(
                "Synthetic Call / Assurance (Long Stock + Long Put).",
                "Préservation du capital. On fixe une perte maximale connue tout en conservant 100% du potentiel de hausse de l'action.",
                "Incertitude à court terme sur une détention long terme (Earnings, Élections) ou marché techniquement fragile."
            ),
            "Short": "N/A"
        },
        "Straddle": {
            "Long": format_desc(
                "Stratégie de Volatilité Pure (Delta Neutre, Long Gamma, Long Vega).",
                "Pari sur l'amplitude du mouvement, indépendamment de la direction. Nécessite que le mouvement réalisé soit supérieur à la volatilité implicite payée.",
                "Événements binaires majeurs : Annonces de résultats (Earnings), décisions de banques centrales, chiffres de l'inflation (CPI)."
            ),
            "Short": format_desc(
                "Vente de Volatilité (Short Gamma, Short Vega).",
                "Pari sur le retour au calme (Mean Reversion) ou la compression de volatilité. Risque élevé si le marché décale.",
                "Marché sans tendance (Range) après un pic de volatilité injustifié."
            )
        },
        "Strangle": {
            "Long": format_desc(
                "Volatilité Pure à Coût Réduit (Long Gamma, Long Vega).",
                "Similaire au Straddle mais moins onéreux car les strikes sont OTM. Nécessite un mouvement plus violent pour atteindre le point mort.",
                "Scénarios 'Cygne Noir' ou ruptures techniques majeures sur des actifs volatils."
            ),
            "Short": format_desc(
                "Vente de Volatilité (Short Gamma, Short Vega).",
                "Encaissement de prime avec une marge d'erreur plus large que le Straddle. Profitable tant que le cours reste entre les deux bornes.",
                "Marché latéral (Sideways market) avec volatilité implicite élevée."
            )
        },
        "Bull Call Spread": {
            "Long": format_desc(
                "Directionnel Haussier à Risque Défini (Vertical Spread).",
                "Réduction du coût de revient par rapport à un Call sec. Le potentiel de gain est plafonné, mais le point mort est plus bas. Exposition réduite à la baisse de volatilité.",
                "Tendance haussière modérée et régulière. Idéal quand la volatilité est trop chère pour acheter un Call simple."
            ),
            "Short": format_desc(
                "Directionnel Baissier (Crédit Spread).",
                "Encaissement d'un crédit. Profitable si le marché baisse, stagne ou monte légèrement (tant qu'il reste sous le strike vendu).",
                "Marché baissier ou résistance technique forte."
            )
        },
        "Bear Put Spread": {
            "Long": format_desc(
                "Directionnel Baissier à Risque Défini (Vertical Spread).",
                "Alternative low-cost à l'achat de Put. On finance l'achat du Put par la vente d'un Put plus bas. Gain capé.",
                "Anticipation d'une baisse mesurée (target précis) plutôt qu'un crash systémique."
            ),
            "Short": format_desc(
                "Directionnel Haussier (Crédit Spread).",
                "Encaissement de crédit (Put Bull Spread). Profitable si le marché monte, stagne ou baisse légèrement.",
                "Marché haussier ou support technique solide (ex: moyenne mobile 200)."
            )
        },
        "Butterfly": {
            "Long": format_desc(
                "Stratégie Neutre / Retour à la Moyenne (Short Gamma, Long Theta).",
                "Capture maximale de la valeur temps (Theta). Le profit est maximal si le cours expire exactement sur le strike central.",
                "Baisse de volatilité attendue. Marché très calme, fin de cycle de mouvement."
            ),
            "Short": format_desc(
                "Volatilité (Long Gamma, Short Theta).",
                "Pari que le cours va sortir d'une zone précise. Risque limité au coût initial.",
                "Sortie de congestion attendue."
            )
        },
        "Call Ratio Backspread": {
            "Long": format_desc(
                "Volatilité Directionnelle Convexe (1 Short ATM / 2 Long OTM).",
                "Gain illimité à la hausse avec un coût d'entrée nul ou négatif (Crédit). Pas de risque à la baisse (sauf légère perte si le cours stagne au strike haut).",
                "**Spécifique Commodities/Tech :** Anticipation d'un 'Spike' violent (Guerre, Pénurie, Rupture techno) avec probabilité de queue de distribution épaisse (Fat Tail)."
            ),
            "Short": format_desc(
                "Contrarian.",
                "Pari que la hausse sera contenue. Très risqué (Naked Calls nets).",
                "Rarement utilisé par les professionnels sous cette forme."
            )
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
        return [("Stock", 0, 1), ("Call", 1.05, -1)] 
    elif strategy == "Protective Put":
        return [("Stock", 0, 1), ("Put", 0.95, 1)] 
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
    elif strategy == "Call Ratio Backspread":
        # Ratio 1x2 pour l'exemple
        return [("Call", 1.0, -1 * pos_mult), ("Call", 1.15, 2 * pos_mult)]
    
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
        st.header("2. Market Data")
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike (K - ATM)", value=100.0)
        T = st.slider("Maturity (Years)", 0.01, 2.0, 0.5, step=0.01)
        sigma = st.slider("Implied Volatility (σ)", 0.05, 2.00, 0.35)
        r = st.number_input("Risk Free Rate (r)", value=0.04)

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
    # --- Bloc Explication PRO ---
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

    st.subheader("Simulateur P&L (Maturity)")
    
    # Range dynamique pour bien voir les Backspreads et autres
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
    
    # Zones colorées
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity >= 0), color='#2E8B57', alpha=0.3, interpolate=True, label="Profit Zone")
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity < 0), color='#CD5C5C', alpha=0.3, interpolate=True, label="Loss Zone")
    
    ax.plot(S_range, pnl_maturity, color="white", linewidth=2.5)
    
    # Lignes de référence
    ax.axhline(0, color='gray', linewidth=1, linestyle='-')
    ax.axvline(S, color='#FFD700', linestyle='--', linewidth=1.5, label=f"Spot Actuel: {S}")
    
    # Breakeven points (approximatifs pour la visu)
    # Simple annotation pour le look pro
    ax.set_title("Profit / Loss Profile at Expiry", color='white', pad=20)
    
    # Design Pro Dark
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

    st.caption("Détail de la structuration (Legs)")
    legs_data = [{"Type": t, "Strike": f"{k:.2f}" if k > 0 else "Mkt", "Qté": q, "Side": "Long" if q > 0 else "Short"} for t, k, q in real_legs_details]
    st.dataframe(legs_data, use_container_width=True)

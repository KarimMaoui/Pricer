import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Derivatives Pricer", layout="wide")
st.markdown("Coded by [Karim MAOUI](https://github.com/KarimMaoui)")

# --- 1. MOTEUR MATHÉMATIQUE ---

def black_scholes(S, K, T, r, sigma, q, option_type="Call"):
    if option_type == "Stock": return S 
    if T <= 1e-5: T = 1e-5
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def get_greeks(S, K, T, r, sigma, q, option_type="Call"):
    if option_type == "Stock": return 1.0, 0.0, 0.0, 0.0 
    if T <= 1e-5: T = 1e-5

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # 1. DELTA
    if option_type == "Call":
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)

    # 2. GAMMA (Identique Call/Put)
    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

    # 3. VEGA (Identique Call/Put, divisé par 100 pour avoir l'impact de 1% de vol)
    vega = (S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)) / 100

    # 4. THETA (Formule Complète Généralisée)
    # Terme 1 : Érosion commune due à la volatilité
    term1 = -(S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
    
    if option_type == "Call":
        term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = + q * S * np.exp(-q * T) * norm.cdf(d1)
    else: # Put
        term2 = + r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = - q * S * np.exp(-q * T) * norm.cdf(-d1)
    
    theta_annual = term1 + term2 + term3
    
    # On divise par 365 pour avoir le Theta "Par Jour" (Standard de marché)
    theta = theta_annual / 365

    return delta, gamma, theta, vega

# --- 2. LOGIQUE DES STRATÉGIES (FUSION EXPERT + NOUVEAUX PRODUITS) ---

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
                "Purchase of a Call option at strike K.",
                "Pure directional leverage strategy. The investor pays a premium to capture 100% of the upside beyond the breakeven point (Strike + Premium), with risk strictly limited to the invested capital.",
                "Bullish market with strong conviction (Momentum). Ideal when implied volatility is low (cheap Call)."
            ),
            "Short": format_desc(
                "Naked sale of a Call at strike K (Naked Call).",
                "Yield strategy. Profit is limited to the received premium. Risk is theoretically unlimited if the underlying explodes upwards. Short Vega position (selling volatility).",
                "Bearish market or hitting a major technical resistance. Anticipation of volatility drop."
            )
        },
        "Put": {
            "Long": format_desc(
                "Purchase of a Put option at strike K.",
                "Protection (Floor) or bearish speculation. The investor sets a guaranteed sell price at K, immunizing against any drop in the underlying below this level.",
                "Portfolio hedging (Hedging) or anticipation of a violent correction."
            ),
            "Short": format_desc(
                "Naked sale of a Put at strike K (Naked Put).",
                "Accumulation strategy (Target Buying). The investor commits to buying the stock at price K. Collects the premium while waiting for the market to drop to their target buy level.",
                "Neutral to slightly bullish market. Willingness to acquire the underlying at a discount (Strike - Received Premium)."
            )
        },
        "Covered Call": {
            "Long": format_desc(
                "Long position on the Underlying + Sale of an OTM Call (Strike K).",
                "Yield Enhancement. The collected premium acts as a synthetic dividend and cushions a slight drop. In exchange, the investor gives up any performance above K.",
                "Neutral or slow bullish market. Ideal for monetizing an existing position when volatility is high."
            ),
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": format_desc(
                "Long position on the Underlying + Purchase of an OTM Put (Strike K).",
                "Total capital insurance (Synthetic Call). The profit profile is unlimited to the upside, but maximum loss is locked at Strike K (minus the cost of insurance).",
                "Short-term uncertainty (Earnings, Elections) on a strategic position one does not wish to sell."
            ),
            "Short": "N/A"
        },
        "Collar": {
            "Long": format_desc(
                "Long Underlying + Buy Put K1 (Protection) + Sell Call K2 (Financing).",
                "Zero Cost Collar. Selling the upside potential (above K2) fully finances the purchase of protection (below K1). P&L is tunneled between K1 and K2.",
                "Prudent wealth management. Desire to hedge against a crash without cash outlay."
            ),
            "Short": "N/A"
        },
        "Risk Reversal": {
            "Long": format_desc(
                "Purchase of an OTM Call (K2) + Sale of an OTM Put (K1).",
                "Funded synthetic Long position. Replicates stock behavior with leverage and no initial capital (if Call Premium = Put Premium). Plays the volatility 'Skew'.",
                "Anticipation of a bullish reversal or volatility arbitrage (Expensive Puts, Cheap Calls)."
            ),
            "Short": format_desc(
                "Sale of an OTM Call (K2) + Purchase of an OTM Put (K1).",
                "Funded synthetic Short position. Allows betting on the downside by financing the Put purchase with the Call sale.",
                "Anticipation of a bearish reversal."
            )
        },
        "Straddle": {
            "Long": format_desc(
                "Buy ATM Call + Buy ATM Put (Same Strike K).",
                "Pure Volatility Strategy (Delta Neutral). Investor wins if the movement amplitude (up or down) exceeds the total premium paid. No directional opinion required.",
                "Before a major binary event (Earnings, CPI, FDA Decision) likely to cause a price Gap."
            ),
            "Short": format_desc(
                "Sell ATM Call + Sell ATM Put (Same Strike K).",
                "Aggressive Volatility Selling. Investor bets the price will remain stuck at Strike K. Max gain is the premium, but risk is unlimited on both sides.",
                "Strict range market. Bet on implied volatility drop (Volatility Crush) after an event."
            )
        },
        "Strangle": {
            "Long": format_desc(
                "Buy OTM Put (K1) + Buy OTM Call (K2).",
                "Low-Cost Volatility. Cheaper than the Straddle, but requires a much more violent movement to reach breakevens.",
                "Bet on an extreme event (Black Swan) or major technical break, with a limited premium budget."
            ),
            "Short": format_desc(
                "Sell OTM Put (K1) + Sell OTM Call (K2).",
                "Volatility Selling with safety margin. Profitable as long as the price stays within the tunnel [K1, K2]. High Probability Trading.",
                "Sideways market without nearby catalyst."
            )
        },
        "Strap": {
            "Long": format_desc(
                "Buy 2 ATM Calls + Buy 1 ATM Put.",
                "Volatility with Bullish Bias. Modified Straddle that doubles down on the upside. If market explodes upwards, gains are multiplied (Net Positive Delta).",
                "Volatility expected but dominant bullish conviction."
            ),
            "Short": format_desc(
                "Sell 2 ATM Calls + Sell 1 ATM Put.",
                "Volatility Selling with Bearish Bias. Very risky: losses accelerate twice as fast if the market rises.",
                "Calm or slow bearish market."
            )
        },
        "Condor": {
            "Long": format_desc(
                "Buy Call K1, Sell Call K2, Sell Call K3, Buy Call K4.",
                "Volatility Arbitrage (Structural Iron Condor). Goal is to capture maximum time value (Theta). Profit is maximal if price ends between K2 and K3.",
                "Perfectly calm market (Index in range)."
            ),
            "Short": format_desc(
                "Sell Call K1, Buy Call K2, Buy Call K3, Sell Call K4.",
                "Breakout Strategy. Betting the price will violently exit the [K2, K3] zone, regardless of direction.",
                "Imminent congestion exit."
            )
        },
        "Bull Call Spread": {
            "Long": format_desc(
                "Buy Call K1 + Sell Call K2.",
                "Optimized bullish exposure. Selling K2 reduces K1 cost, lowering the breakeven. In exchange, profit is capped at K2. Risk/Reward ratio often superior to naked Call.",
                "Moderate rise anticipated towards a specific target (K2)."
            ),
            "Short": format_desc(
                "Sell Call K1 + Buy Call K2 (Credit Spread).",
                "Bearish Credit Strategy. Betting the market won't exceed K1. Gain limited to initial credit received.",
                "Bearish trend or strong technical resistance at K1."
            )
        },
        "Bear Put Spread": {
            "Long": format_desc(
                "Buy Put K2 + Sell Put K1.",
                "Optimized bearish exposure. Selling K1 reduces K2 Put cost. Profit is maximal if price reaches K1.",
                "Moderate drop anticipated towards a specific support (K1)."
            ),
            "Short": format_desc(
                "Sell Put K2 + Buy Put K1 (Credit Spread).",
                "Bullish Credit Strategy (Bull Put Spread). Betting the market won't drop below K2. Collecting the premium.",
                "Bullish trend or solid technical support at K2."
            )
        },
        "Seagull": {
            "Long": format_desc(
                "Bull Call Spread (Buy K2 / Sell K3) + Sell Put K1.",
                "Structured construction. Upside is financed by selling Put K1 and Call K3. Often structured at 'Zero Cost'. Risk is shifted to the downside below K1.",
                "Bullish market, with willingness to buy the underlying in case of major pullback to level K1 (Target Buying)."
            ),
            "Short": "N/A"
        },
        "Butterfly": {
            "Long": format_desc(
                "Buy Call K1 + Sell 2 Calls K2 + Buy Call K3.",
                "Volatility Sniper (Short Gamma). Max profit highly localized on central strike K2. Low Risk, High Reward, Low Probability.",
                "Anemic market, approaching expiration (Pin Risk)."
            ),
            "Short": "N/A"
        },
        "Call Ratio Backspread": {
            "Long": format_desc(
                "Sell 1 ATM Call (K1) + Buy 2 OTM Calls (K2).",
                "Convex Volatility Strategy. Selling Call K1 finances multiple purchases of K2. If market explodes, gain is unlimited (Delta becomes very positive). Loss risk only if market stagnates at K2.",
                "Bullish explosion expected (Commodities, Tech). Aiming to 'Gamma Scalp' the rise."
            ),
            "Short": "N/A"
        },
        "Put Ratio Backspread": {
            "Long": format_desc(
                "Sell 1 ATM Put (K2) + Buy 2 OTM Puts (K1).",
                "Anti-Crash Protection. Free or credit structure generating massive profit if market collapses. Loss risk limited to zone between K1 and K2.",
                "Portfolio hedging against systemic risk (Black Swan)."
            ),
            "Short": "N/A"
        },
        "Synthetic Long": {
            "Long": format_desc(
                "Buy ATM Call + Sell ATM Put.",
                "Delta One Replication. This combination offers exactly the same gain/loss profile as holding the stock, but with near-zero tied capital (excluding margin).",
                "Desire for linear exposure without available cash."
            ),
            "Short": format_desc(
                "Sell ATM Call + Buy ATM Put.",
                "Synthetic Short Position. Allows shorting the market without borrowing shares (Hard-to-borrow stocks).",
                "Pure bearish conviction."
            )
        }
    }
    return desc.get(strategy, {}).get(position, "N/A")

# --- 3. ANALYSE DES RISQUES (VERSION TECHNIQUE) ---

def get_greeks_profile(strategy, position):
    # Tuple : (Delta, Gamma, Theta, Vega)
    profiles = {
        "Call": {
            "Long": ("Positive. Delta = Approximate probability of exercise.", "Positive. Max acceleration ATM.", "Negative. The option is a wasting asset.", "Positive. Max Vega ATM."),
            "Short": ("Negative. You are against the market.", "Negative. Risk of 'Gap' against you.", "Positive. You collect time value.", "Negative. Vol drop reduces your buyback cost.")
        },
        "Put": {
            "Long": ("Negative. Delta tends to -1 if ITM.", "Positive. Option becomes more sensitive if market drops.", "Negative. Cost of carry.", "Positive. Put gains value if fear rises."),
            "Short": ("Positive. Delta tends to 0 if OTM.", "Negative. Accelerated risk to the downside.", "Positive. Daily income.", "Negative. You are selling insurance.")
        },
        "Covered Call": {
            "Long": ("Reduced Positive. Short Call K2 acts as a drag on Stock Delta (1.0).", "Negative. Short Call Gamma dominates (Stock has 0 Gamma).", "Positive. Only the Short Call leg generates Theta.", "Negative. If Vol rises, sold Call becomes more expensive to buy back."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Protective Put": {
            "Long": ("Positive. Put (-Delta) reduces Stock exposure.", "Positive. Put adds downside convexity (Cushion).", "Negative. Net cost of insurance.", "Positive. Your protection is worth more if vol rises."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Collar": {
            "Long": ("Positive. Capped to the upside (Call) and downside (Put).", "Variable. Long Gamma on Put (Low), Short Gamma on Call (High).", "Mixed. Depends on premiums. Generally Positive Theta near sold Call.", "Negative (Generally). Skew (Put Vol > Call Vol) makes Call sale less sensitive than Put purchase."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Straddle": {
            "Long": ("Neutral (if ATM). Dynamic adjustment required.", "Strong Positive. Sum of both options. Max ATM.", "Strong Negative. High cost of carry.", "Strong Positive. Maximum exposure to Volatility."),
            "Short": ("Neutral.", "Strong Negative. Immediate danger on gap.", "Strong Positive. Max time decay gain.", "Strong Negative. Pure Short Vol.")
        },
        "Strangle": {
            "Long": ("Neutral.", "Positive. Lower than Straddle (OTM Strikes).", "Negative. Cheaper than Straddle.", "Positive. Vega sensitivity present."),
            "Short": ("Neutral.", "Negative.", "Positive.", "Negative.")
        },
        "Strap": {
            "Long": ("Positive (Bullish Bias). 2 Calls vs 1 Put.", "Strong Positive. Massive Gamma.", "Strong Negative. 3 premiums to pay.", "Strong Positive. Tripled Vega exposure."),
            "Short": ("Negative (Bearish Bias).", "Strong Negative.", "Strong Positive.", "Strong Negative.")
        },
        "Bull Call Spread": {
            "Long": ("Positive. Net Long Delta (Buy K1 > Sell K2).", "Gamma Flip. Long Gamma low (K1), Short Gamma high (K2).", "Mixed. Paying time on K1, receiving on K2.", "Mixed. Long Vega on K1, Short on K2. Sensitive to term structure."),
            "Short": ("Negative.", "Inverse of Long: Short Gamma low, Long Gamma high.", "Mixed.", "Mixed.")
        },
        "Bear Put Spread": {
            "Long": ("Negative. Net Short Delta.", "Gamma Flip. Short Gamma low (K1), Long Gamma high (K2).", "Mixed. Receiving time on K1, paying on K2.", "Mixed. Short Vega on K1, Long on K2."),
            "Short": ("Positive.", "Inverse of Long.", "Mixed.", "Mixed.")
        },
        "Condor": {
            "Long": ("Neutral.", "Negative (Short Gamma) on central plateau.", "Positive. Max time gain on plateau.", "Negative. Short Volatility."),
            "Short": ("Neutral.", "Positive (Long Gamma) in center.", "Negative. Cost of time.", "Positive. Long Volatility.")
        },
        "Seagull": {
            "Long": ("Positive. Upside leverage (Call Spread).", "Negative (Overall). Short Gamma on the 2 sold strikes (Low Put + High Call).", "Positive. Both sales largely finance purchased Call Theta.", "Negative. Selling 2 legs (Put + Call) vs 1 buy. Net Short Vega."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Butterfly": {
            "Long": ("Neutral.", "Negative at center (Massive Short Gamma ATM).", "Strong Positive. Max Theta ATM (Time is your ally).", "Negative. Short Volatility ATM. You want Vol to crash."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Call Ratio Backspread": {
            "Long": ("Variable (Often Positive).", "Strong Positive. Convexity of 2 long Calls dominates single sale.", "Negative. You have 2 options losing time value.", "Positive. Quantity: 2x Vega OTM > 1x Vega ATM."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Put Ratio Backspread": {
            "Long": ("Variable (Often Negative).", "Strong Positive. Net downside convexity.", "Negative.", "Positive. Quantity: 2x Vega OTM > 1x Vega ATM."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Risk Reversal": {
            "Long": ("Positive. Sum of Deltas (Long Call + Short Put).", "Neutral (Linear). Gammas cancel out or are far.", "Neutral.", "Variable. Depends on Skew (Put Vol vs Call Vol)."),
            "Short": ("Negative.", "Neutral.", "Neutral.", "Variable.")
        },
        "Synthetic Long": {
            "Long": ("Positive (100%). Fixed Delta of 1.0.", "Neutral (0). Call and Put Gammas cancel out.", "Neutral (0).", "Neutral (0). Call and Put Vegas cancel out."),
            "Short": ("Negative (100%).", "Neutral.", "Neutral.", "Neutral.")
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
            st.caption("No spread parameters for this strategy")
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
        
        # SÉLECTEUR DE MODÈLE
        model_choice = st.radio("Modèle / Model", ["Equity (Black-Scholes)", "Commodity (Black-76)"], horizontal=True)
        
        # Label dynamique
        label_S = "Future Price (F)" if "Black-76" in model_choice else "Spot Price (S)"
        S = st.number_input(label_S, value=100.0)
        
        K = st.number_input("Strike Central (K)", value=100.0)
        T = st.slider("Maturity (Years)", 0.01, 5.0, 1.0, step=0.01)
        sigma = st.slider("Implied Volatility (ATM)", 0.01, 1.50, 0.30, step=0.01)
        
        # GESTION DU SKEW
        enable_skew = st.checkbox("Enable Volatility Skew")
        skew_vol = 0.0
        if enable_skew:
            st.caption("Skew positif = Calls plus chers. Skew négatif = Puts plus chers.")
            skew_vol = st.slider("Skew (Vol Call - Vol Put)", -0.20, 0.20, 0.00, step=0.01)
            
        r = st.number_input("Risk Free Rate (r)", value=0.03)

        # --- AJOUT: GESTION DES DIVIDENDES (Seulement pour Equity) ---
        div_yield = 0.0
        if "Equity" in model_choice:
            enable_div = st.checkbox("Include Dividends")
            if enable_div:
                div_yield = st.number_input("Dividend Yield (q)", value=0.02, step=0.001, format="%.3f")
# Calculs
legs_config = get_strategy_legs(selected_strat, K, width_lower, width_upper, position)
total_price, total_delta, total_gamma, total_theta, total_vega = 0, 0, 0, 0, 0
real_legs_details = []

# --- DÉFINITION DU DIVIDENDE (q) ---
if "Black-76" in model_choice:
    q_model = r          # En Commo (Futures), q = r (Cost of Carry)
else:
    q_model = div_yield  # En Equity, q = Dividende (0 par défaut)

for leg_type, strike_mult, qty in legs_config:
    leg_k = K * strike_mult if leg_type != "Stock" else 0
    
    # --- SKEW LOGIC ---
    leg_sigma = sigma
    if enable_skew and leg_type != "Stock":
        if leg_type == "Call":
            leg_sigma = sigma + (skew_vol / 2)
        elif leg_type == "Put":
            leg_sigma = sigma - (skew_vol / 2)
    
    # Pricing (Le moteur reçoit le bon q_model)
    p = black_scholes(S, leg_k, T, r, leg_sigma, q_model, leg_type)
    d, g, t, v = get_greeks(S, leg_k, T, r, leg_sigma, q_model, leg_type)
    
    total_price += p * qty
    total_delta += d * qty
    total_gamma += g * qty
    total_theta += t * qty
    total_vega += v * qty
    
    # On stocke les 4 valeurs (avec le _ pour ignorer la vol dans les boucles d'affichage graphiques)
    real_legs_details.append((leg_type, leg_k, qty, leg_sigma))

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
    
    for leg_type, leg_k, qty, _ in real_legs_details:
        if leg_type == "Call": pnl_maturity += np.maximum(S_range - leg_k, 0) * qty
        elif leg_type == "Put": pnl_maturity += np.maximum(leg_k - S_range, 0) * qty
        elif leg_type == "Stock": pnl_maturity += S_range * qty

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity >= 0), color='#2E8B57', alpha=0.3, interpolate=True)
    ax.fill_between(S_range, pnl_maturity, 0, where=(pnl_maturity < 0), color='#CD5C5C', alpha=0.3, interpolate=True)
    ax.plot(S_range, pnl_maturity, color="white", linewidth=2.5)
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(S, color='#FFD700', linestyle='--', label=f"Spot: {S}")
    
    if selected_strat in ["Call", "Put"]:
        strike_plot = real_legs_details[0][1]
        y_max = ax.get_ylim()[1]
        ax.text(strike_plot, y_max * 0.95, "ATM", color='#FFD700', ha='center', fontweight='bold')
        if selected_strat == "Call":
            ax.text(strike_plot * 0.85, y_max * 0.85, "OTM", color='cyan', ha='center', fontsize=10, alpha=0.8)
            ax.text(strike_plot * 1.15, y_max * 0.85, "ITM", color='cyan', ha='center', fontsize=10, alpha=0.8)
        else: # Put
            ax.text(strike_plot * 0.85, y_max * 0.85, "ITM", color='cyan', ha='center', fontsize=10, alpha=0.8)
            ax.text(strike_plot * 1.15, y_max * 0.85, "OTM", color='cyan', ha='center', fontsize=10, alpha=0.8)

    for t, k, q, _ in real_legs_details: 
        if k > 0:
            ax.axvline(k, color='gray', linestyle=':', alpha=0.5)
            if selected_strat not in ["Call", "Put"]:
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
    # On ajoute la colonne "Vol"
    legs_data = [{"Type": t, "Strike": f"{k:.2f}" if k > 0 else "Mkt", "Qté": q, "Side": "Long" if q > 0 else "Short", "Vol Utilise": f"{vol:.1%}"} for t, k, q, vol in real_legs_details]
    st.dataframe(legs_data, use_container_width=True)

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


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
    
    delta = np.exp(-q * T) * (norm.cdf(d1) if option_type == "Call" else -norm.cdf(-d1))
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100
    theta = (- (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))) / 365
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
                "Achat d'un Call de strike K.",
                "Stratégie de levier directionnel pur. L'investisseur paie une prime pour capter 100% de la hausse au-delà du point mort (Strike + Prime), avec un risque strictement limité au capital investi.",
                "Marché haussier avec forte conviction (Momentum). Idéal quand la volatilité implicite est faible (Call pas cher)."
            ),
            "Short": format_desc(
                "Vente à découvert d'un Call de strike K (Naked Call).",
                "Stratégie de rendement (Yield). Le gain est limité à la prime reçue. Le risque est théoriquement illimité si le sous-jacent explose à la hausse. Position Short Vega (on vend de la volatilité).",
                "Marché baissier ou butée sur une résistance technique majeure. Anticipation d'une baisse de la volatilité."
            )
        },
        "Put": {
            "Long": format_desc(
                "Achat d'un Put de strike K.",
                "Protection (Floor) ou spéculation baissière. L'investisseur fixe un prix de vente garanti à K, s'immunisant contre toute baisse du sous-jacent sous ce niveau.",
                "Couverture de portefeuille (Hedging) ou anticipation d'une correction violente."
            ),
            "Short": format_desc(
                "Vente à découvert d'un Put de strike K (Naked Put).",
                "Stratégie d'accumulation (Target Buying). L'investisseur s'engage à acheter le titre au cours K. Il encaisse la prime en attendant que le marché baisse vers son niveau d'achat cible.",
                "Marché neutre à légèrement haussier. Volonté d'acquérir le sous-jacent avec un rabais (Strike - Prime reçue)."
            )
        },
        "Covered Call": {
            "Long": format_desc(
                "Position Longue sur le Sous-jacent + Vente d'un Call OTM (Strike K).",
                "Amélioration du rendement (Yield Enhancement). La prime encaissée agit comme un dividende synthétique et amortit une légère baisse. En contrepartie, l'investisseur renonce à toute performance au-delà de K.",
                "Marché neutre ou haussier lent. Idéal pour monétiser une position existante quand la volatilité est élevée."
            ),
            "Short": "N/A"
        },
        "Protective Put": {
            "Long": format_desc(
                "Position Longue sur le Sous-jacent + Achat d'un Put OTM (Strike K).",
                "Assurance totale du capital (Synthetic Call). Le profil de gain est illimité à la hausse, mais la perte maximale est verrouillée au niveau du Strike K (moins le coût de l'assurance).",
                "Incertitude à court terme (Earnings, Élections) sur une position stratégique qu'on ne souhaite pas solder."
            ),
            "Short": "N/A"
        },
        "Collar": {
            "Long": format_desc(
                "Long Sous-jacent + Achat Put K1 (Protection) + Vente Call K2 (Financement).",
                "Protection à coût nul (Zero Cost Collar). La vente du potentiel de hausse (au-dessus de K2) finance intégralement l'achat de la protection (en-dessous de K1). Le P&L est tunellisé entre K1 et K2.",
                "Gestion de fortune prudente. Volonté de se couvrir contre un krach sans débourser de trésorerie."
            ),
            "Short": "N/A"
        },
        "Risk Reversal": {
            "Long": format_desc(
                "Achat d'un Call OTM (K2) + Vente d'un Put OTM (K1).",
                "Position synthétique Longue financée. Réplique le comportement de l'action avec un effet de levier et sans capital initial (si Prime Call = Prime Put). On joue le 'Skew' de volatilité.",
                "Anticipation d'un retournement haussier (Reversal) ou arbitrage de volatilité (Puts chers, Calls donnés)."
            ),
            "Short": format_desc(
                "Vente d'un Call OTM (K2) + Achat d'un Put OTM (K1).",
                "Position synthétique Courte financée. Permet de parier sur la baisse en finançant l'achat du Put par la vente du Call.",
                "Anticipation d'un retournement baissier."
            )
        },
        "Straddle": {
            "Long": format_desc(
                "Achat Call ATM + Achat Put ATM (Même Strike K).",
                "Stratégie de Volatilité Pure (Delta Neutre). L'investisseur gagne si l'amplitude du mouvement (hausse ou baisse) dépasse la prime totale payée. Aucune opinion directionnelle requise.",
                "Avant un événement binaire majeur (Earnings, CPI, Décision FDA) susceptible de provoquer un Gap de prix."
            ),
            "Short": format_desc(
                "Vente Call ATM + Vente Put ATM (Même Strike K).",
                "Vente de Volatilité agressive. L'investisseur parie que le cours restera collé au Strike K. Le gain maximal est la prime, mais le risque est illimité des deux côtés.",
                "Marché en range strict. Pari sur une baisse de la volatilité implicite (Volatility Crush) après un événement."
            )
        },
        "Strangle": {
            "Long": format_desc(
                "Achat Put OTM (K1) + Achat Call OTM (K2).",
                "Volatilité à coût réduit. Moins cher que le Straddle, mais nécessite un mouvement beaucoup plus violent pour atteindre les points morts (Breakevens).",
                "Pari sur un événement extrême (Cygne Noir) ou une rupture technique majeure, avec un budget prime limité."
            ),
            "Short": format_desc(
                "Vente Put OTM (K1) + Vente Call OTM (K2).",
                "Vente de Volatilité avec marge de sécurité. Profitable tant que le cours reste dans le tunnel [K1, K2]. Probabilité de gain élevée (High Probability Trading).",
                "Marché latéral (Sideways) sans catalyseur proche."
            )
        },
        "Strap": {
            "Long": format_desc(
                "Achat de 2 Calls ATM + Achat d'1 Put ATM.",
                "Volatilité avec Biais Haussier. C'est un Straddle modifié qui double la mise à la hausse. Si le marché explose à la hausse, les gains sont démultipliés (Delta positif net).",
                "Volatilité attendue mais conviction haussière dominante."
            ),
            "Short": format_desc(
                "Vente de 2 Calls ATM + Vente d'1 Put ATM.",
                "Vente de Volatilité avec Biais Baissier. Très risqué : les pertes s'accélèrent deux fois plus vite si le marché monte.",
                "Marché calme ou baissier lent."
            )
        },
        "Condor": {
            "Long": format_desc(
                "Achat Call K1, Vente Call K2, Vente Call K3, Achat Call K4.",
                "Arbitrage de Volatilité (Iron Condor structurel). L'objectif est de capturer la valeur temps (Theta) maximale. Le profit est maximal si le cours termine entre K2 et K3.",
                "Marché parfaitement calme (Indice en range)."
            ),
            "Short": format_desc(
                "Vente Call K1, Achat Call K2, Achat Call K3, Vente Call K4.",
                "Stratégie de Breakout. On parie que le cours va sortir violemment de la zone [K2, K3], peu importe la direction.",
                "Sortie de congestion imminente."
            )
        },
        "Bull Call Spread": {
            "Long": format_desc(
                "Achat Call K1 + Vente Call K2.",
                "Exposition haussière optimisée. La vente de K2 réduit le coût du K1, abaissant ainsi le point mort. En échange, le profit est capé à K2. Ratio Risque/Rendement souvent supérieur au Call sec.",
                "Hausse modérée anticipée vers une cible précise (K2)."
            ),
            "Short": format_desc(
                "Vente Call K1 + Achat Call K2 (Credit Spread).",
                "Stratégie de Crédit baissière. On parie que le marché ne dépassera pas K1. Le gain est limité au crédit reçu initialement.",
                "Tendance baissière ou résistance technique forte à K1."
            )
        },
        "Bear Put Spread": {
            "Long": format_desc(
                "Achat Put K2 + Vente Put K1.",
                "Exposition baissière optimisée. La vente de K1 réduit le coût du Put K2. Le profit est maximal si le cours atteint K1.",
                "Baisse modérée anticipée vers un support précis (K1)."
            ),
            "Short": format_desc(
                "Vente Put K2 + Achat Put K1 (Credit Spread).",
                "Stratégie de Crédit haussière (Bull Put Spread). On parie que le marché ne passera pas sous K2. On encaisse la prime.",
                "Tendance haussière ou support technique solide à K2."
            )
        },
        "Seagull": {
            "Long": format_desc(
                "Bull Call Spread (Achat K2 / Vente K3) + Vente Put K1.",
                "Construction structurée. La hausse est financée par la vente du Put K1 et du Call K3. Souvent structuré à 'Zéro Prime'. Le risque est reporté à la baisse sous K1.",
                "Marché haussier, avec une volonté d'acheter le sous-jacent en cas de repli majeur au niveau K1 (Target Buying)."
            ),
            "Short": "N/A"
        },
        "Butterfly": {
            "Long": format_desc(
                "Achat Call K1 + Vente 2 Calls K2 + Achat Call K3.",
                "Sniper de Volatilité (Short Gamma). Profit maximal très localisé sur le strike central K2. Risque faible, Reward élevé, Probabilité faible.",
                "Marché anémique, expiration proche (Pin Risk)."
            ),
            "Short": "N/A"
        },
        "Call Ratio Backspread": {
            "Long": format_desc(
                "Vente 1 Call ATM (K1) + Achat 2 Calls OTM (K2).",
                "Stratégie de Volatilité Convexe. La vente du Call K1 finance l'achat multiple de K2. Si le marché explose, le gain est illimité (Delta devient très positif). Risque de perte uniquement si le marché stagne à K2.",
                "Explosion haussière attendue (Matières Premières, Tech). On veut 'Gamma Scalper' la hausse."
            ),
            "Short": "N/A"
        },
        "Put Ratio Backspread": {
            "Long": format_desc(
                "Vente 1 Put ATM (K2) + Achat 2 Puts OTM (K1).",
                "Protection Anti-Krach. Structure gratuite ou à crédit qui génère un profit massif en cas d'effondrement du marché. Risque de perte limité à la zone entre K1 et K2.",
                "Couverture de portefeuille contre un risque systémique (Black Swan)."
            ),
            "Short": "N/A"
        },
        "Synthetic Long": {
            "Long": format_desc(
                "Achat Call ATM + Vente Put ATM.",
                "Réplication Delta One. Cette combinaison offre exactement le même profil de gain/perte que la détention de l'action, mais avec un capital immobilisé quasi-nul (hors marge).",
                "Volonté d'exposition linéaire sans trésorerie disponible."
            ),
            "Short": format_desc(
                "Vente Call ATM + Achat Put ATM.",
                "Position Courte Synthétique. Permet de shorter le marché sans emprunter les titres (Hard-to-borrow stocks).",
                "Conviction baissière pure."
            )
        }
    }
    return desc.get(strategy, {}).get(position, "N/A")

# --- 3. ANALYSE DES RISQUES (VERSION TECHNIQUE) ---

def get_greeks_profile(strategy, position):
    # Tuple : (Delta, Gamma, Theta, Vega)
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
            "Long": ("Positif Réduit. Le Short Call K2 freine le Delta du stock (1.0).", "Négatif. Le Gamma du Short Call domine (le Stock a un Gamma de 0).", "Positif. Seule la jambe Short Call génère du Theta.", "Négatif. Si la Vol monte, le Call vendu devient plus cher à racheter."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Protective Put": {
            "Long": ("Positif. Le Put (-Delta) réduit l'exposition du Stock.", "Positif. Le Put ajoute de la convexité à la baisse (Coussin).", "Négatif. Coût net de l'assurance.", "Positif. Votre protection vaut plus cher si la vol monte."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Collar": {
            "Long": ("Positif. Limité à la hausse (Call) et à la baisse (Put).", "Variable. Long Gamma sur le Put (Bas), Short Gamma sur le Call (Haut).", "Mixte. Dépend des primes. Généralement Theta Positif proche du Call vendu.", "Négatif (Généralement). Le Skew (Vol Put > Vol Call) rend la vente du Call moins sensible que l'achat du Put."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Straddle": {
            "Long": ("Neutre (si ATM). Ajustement dynamique requis.", "Positif Fort. Cumul des deux options. Max ATM.", "Négatif Fort. Coût de portage élevé.", "Positif Fort. Exposition maximale à la Volatilité."),
            "Short": ("Neutre.", "Négatif Fort. Danger immédiat sur écart.", "Positif Fort. Gain temps maximal.", "Négatif Fort. Short Vol pur.")
        },
        "Strangle": {
            "Long": ("Neutre.", "Positif. Plus faible que le Straddle (Strikes OTM).", "Négatif. Moins coûteux que le Straddle.", "Positif. Sensibilité Vega présente."),
            "Short": ("Neutre.", "Négatif.", "Positif.", "Négatif.")
        },
        "Strap": {
            "Long": ("Positif (Biais Haussier). 2 Calls vs 1 Put.", "Positif Fort. Gamma massif.", "Négatif Fort. 3 primes à payer.", "Positif Fort. Exposition Vega triplée."),
            "Short": ("Négatif (Biais Baissier).", "Négatif Fort.", "Positif Fort.", "Négatif Fort.")
        },
        "Bull Call Spread": {
            "Long": ("Positif. Net Long Delta (Achat K1 > Vente K2).", "Flip de Gamma. Long Gamma en bas (K1), Short Gamma en haut (K2).", "Mixte. Vous payez du temps sur K1, vous en recevez sur K2.", "Mixte. Long Vega sur K1, Short sur K2. Sensible à la structure par terme."),
            "Short": ("Négatif.", "Inverse du Long : Short Gamma bas, Long Gamma haut.", "Mixte.", "Mixte.")
        },
        "Bear Put Spread": {
            "Long": ("Négatif. Net Short Delta.", "Flip de Gamma. Short Gamma en bas (K1), Long Gamma en haut (K2).", "Mixte. Réception de temps sur K1, paiement sur K2.", "Mixte. Short Vega sur K1, Long sur K2."),
            "Short": ("Positif.", "Inverse du Long.", "Mixte.", "Mixte.")
        },
        "Condor": {
            "Long": ("Neutre.", "Négatif (Short Gamma) sur le plateau central.", "Positif. Gain temps maximal sur le plateau.", "Négatif. Short Volatilité."),
            "Short": ("Neutre.", "Positif (Long Gamma) au centre.", "Négatif. Coût du temps.", "Positif. Long Volatilité.")
        },
        "Seagull": {
            "Long": ("Positif. Levier à la hausse (Call Spread).", "Négatif (Globalement). Short Gamma sur les 2 bornes vendues (Put bas + Call haut).", "Positif. Les deux ventes financent largement le Theta du Call acheté.", "Négatif. Vente de 2 pattes (Put + Call) contre 1 achat. Net Short Vega."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Butterfly": {
            "Long": ("Neutre.", "Négatif au centre (Short Gamma massif ATM).", "Positif Fort. Theta Max ATM (Le temps est votre allié).", "Négatif. Short Volatilité ATM. Vous voulez que la Vol s'effondre."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Call Ratio Backspread": {
            "Long": ("Variable (Souvent Positif).", "Positif Fort. La convexité des 2 Calls achetés domine la vente unique.", "Négatif. Vous avez 2 options qui perdent de la valeur temps.", "Positif. Quantité : 2x Vega OTM > 1x Vega ATM."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Put Ratio Backspread": {
            "Long": ("Variable (Souvent Négatif).", "Positif Fort. Convexité nette à la baisse.", "Négatif.", "Positif. Quantité : 2x Vega OTM > 1x Vega ATM."),
            "Short": ("N/A", "N/A", "N/A", "N/A")
        },
        "Risk Reversal": {
            "Long": ("Positif. Cumul des Deltas (Long Call + Short Put).", "Neutre (Linéaire). Les Gamma s'annulent ou sont loin.", "Neutre.", "Variable. Dépend du Skew (Vol Put vs Vol Call)."),
            "Short": ("Négatif.", "Neutre.", "Neutre.", "Variable.")
        },
        "Synthetic Long": {
            "Long": ("Positif (100%). Delta fixe de 1.0.", "Neutre (0). Gamma Call et Put s'annulent.", "Neutre (0).", "Neutre (0). Vega Call et Put s'annulent."),
            "Short": ("Négatif (100%).", "Neutre.", "Neutre.", "Neutre.")
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
        
        # --- AJOUT: SÉLECTEUR DE MODÈLE (Equity vs Commodity) ---
        model_choice = st.radio("Modèle / Model", ["Equity (Black-Scholes)", "Commodity (Black-76)"], horizontal=True)
        
        # --- AJOUT: Label dynamique (Spot vs Future) ---
        label_S = "Future Price (F)" if "Black-76" in model_choice else "Spot Price (S)"
        S = st.number_input(label_S, value=100.0)
        
        K = st.number_input("Strike Central (K)", value=100.0)
        T = st.slider("Maturity (Years)", 0.01, 5.0, 1.0, step=0.01)
        sigma = st.slider("Implied Volatility (ATM)", 0.01, 1.50, 0.30, step=0.01)
        
        # --- AJOUT: GESTION DU SKEW ---
        enable_skew = st.checkbox("Activer Volatility Skew")
        skew_vol = 0.0
        if enable_skew:
            st.caption("Skew positif = Calls plus chers (Agri). Skew négatif = Puts plus chers (Equity/Oil).")
            skew_vol = st.slider("Skew (Vol Call - Vol Put)", -0.20, 0.20, 0.00, step=0.01)
            
        r = st.number_input("Risk Free Rate (r)", value=0.04)

# Calculs
legs_config = get_strategy_legs(selected_strat, K, width_lower, width_upper, position)
total_price, total_delta, total_gamma, total_theta, total_vega = 0, 0, 0, 0, 0
real_legs_details = []

# --- AJOUT: Variable q dynamique (0 pour Equity, r pour Commodity) ---
q_model = r if "Black-76" in model_choice else 0.0

for leg_type, strike_mult, qty in legs_config:
    leg_k = K * strike_mult if leg_type != "Stock" else 0
    
    # --- AJOUT: LOGIQUE DU SKEW ---
    # Si Skew = -10% (-0.10), les Puts ont +5% de vol, les Calls ont -5% de vol.
    leg_sigma = sigma
    if enable_skew and leg_type != "Stock":
        if leg_type == "Call":
            leg_sigma = sigma + (skew_vol / 2)
        elif leg_type == "Put":
            leg_sigma = sigma - (skew_vol / 2)
    
    # Pricing (On utilise leg_sigma au lieu de sigma)
    p = black_scholes(S, leg_k, T, r, leg_sigma, q_model, leg_type)
    d, g, t, v = get_greeks(S, leg_k, T, r, leg_sigma, q_model, leg_type)
    
    total_price += p * qty
    total_delta += d * qty
    total_gamma += g * qty
    total_theta += t * qty
    total_vega += v * qty
    
    # On stocke la volatilité utilisée pour l'afficher plus tard si besoin
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

st.write("---")
st.markdown("Coded by [Karim MAOUI](https://github.com/KarimMaoui)")

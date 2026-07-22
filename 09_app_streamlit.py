import streamlit as st
import pandas as pd
import pickle
import re
import io
import base64
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
from collections import Counter

st.set_page_config(
    page_title="Harmonisation PAA",
    page_icon="⚓",
    layout="wide"
)

def get_logo_base64():
    try:
        with open("logo_paa.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

logo_b64 = get_logo_base64()

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap');

    .main { background-color: #0D1B2A; }
    .stApp { background-color: #0D1B2A; }
    h1, h2, h3 { color: #F0F4F8; font-family: 'Inter', sans-serif; }

    .header-zone {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 20px 32px 20px;
        text-align: center;
        border-bottom: 1px solid #1E3A5F;
        margin-bottom: 32px;
        background: linear-gradient(180deg, #0D1B2A 0%, #1B2B3B 100%);
        border-radius: 0 0 16px 16px;
    }
    .logo-ring {
        width: 220px;
        height: 220px;
        border-radius: 50%;
        border: 3px solid #C8A96E;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 24px auto;
        background: radial-gradient(circle, #1B2B3B 0%, #0D1B2A 100%);
        box-shadow: 0 0 40px rgba(200,169,110,0.25), 0 0 80px rgba(74,144,217,0.1);
        overflow: hidden;
        padding: 0;
    }
    .header-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #C8A96E;
        margin-bottom: 12px;
    }
    .header-titre {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(28px, 4vw, 48px);
        font-weight: 700;
        color: #F0F4F8;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    .header-sous-titre {
        font-family: 'Inter', sans-serif;
        font-size: clamp(14px, 2vw, 18px);
        color: #4A90D9;
        margin-bottom: 16px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .header-desc {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: #8FA8C0;
        max-width: 520px;
        line-height: 1.7;
        margin: 0 auto 28px auto;
        font-weight: 300;
    }
    .stats-bar {
        display: flex;
        gap: 40px;
        justify-content: center;
        margin-top: 24px;
        padding-top: 24px;
        border-top: 1px solid #1E3A5F;
        width: 100%;
    }
    .stat-item { text-align: center; }
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 30px;
        font-weight: 700;
        color: #C8A96E;
        display: block;
    }
    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #8FA8C0;
        margin-top: 4px;
        display: block;
    }
    .divider-line {
        width: 1px;
        height: 40px;
        background: #1E3A5F;
        margin: auto 0;
    }
    .resultat {
        background: linear-gradient(135deg, #1B2B3B 0%, #162436 100%);
        padding: 20px 24px;
        border-left: 4px solid #C8A96E;
        border-radius: 4px;
        margin-top: 15px;
    }
    .resultat h3 { color: #C8A96E; margin: 0; font-family: 'Inter', sans-serif; font-size: 20px; }
    .resultat p  { color: #8FA8C0; margin: 8px 0 0 0; font-family: 'Inter', sans-serif; }

    .stButton button {
        background: linear-gradient(135deg, #1A5276 0%, #2E86C1 100%);
        color: #F0F4F8;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 10px 20px;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #2E86C1 0%, #4A90D9 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74,144,217,0.3);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1B2B3B;
        color: #8FA8C0;
        border-radius: 4px 4px 0 0;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        border: 1px solid #1E3A5F;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1A5276 0%, #2E86C1 100%) !important;
        color: #F0F4F8 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stTextInput input {
        background-color: #1B2B3B !important;
        color: #F0F4F8 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 4px !important;
    }
    .stTextInput input:focus {
        border-color: #4A90D9 !important;
        box-shadow: 0 0 0 2px rgba(74,144,217,0.2) !important;
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0D1B2A; }
    ::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #4A90D9; }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT
# ============================================================
@st.cache_resource
def charger_modele():
    with open("modele_random_forest.pkl", 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def charger_base():
    df_base = pd.read_excel("base_nettoyee.xlsx")
    vectorizer = TfidfVectorizer()
    tous_les_noms = pd.concat([df_base['nom_clean'], df_base['desc_clean']]).fillna('')
    vectorizer.fit(tous_les_noms)
    noms_officiels = df_base[['nom_clean', 'Nom_chargeurs']].drop_duplicates()
    noms_officiels_list = noms_officiels['nom_clean'].dropna().unique().tolist()
    return df_base, vectorizer, noms_officiels, noms_officiels_list

modele = charger_modele()
df_base, vectorizer, noms_officiels, noms_officiels_list = charger_base()


# ============================================================
# FONCTIONS
# ============================================================
def nettoyer_nom(nom):
    if pd.isna(nom) or nom == "":
        return ""
    nom = str(nom).lower()
    nom = nom.replace("s.a.r.l.", "sarl").replace("s.a.r.l", "sarl")
    nom = nom.replace("s.a.", "sa").replace("s.a", "sa")
    nom = nom.replace("cie.", "cie").replace("gie.", "gie")
    nom = nom.replace("b.p.", "bp").replace("b.p", "bp")
    nom = re.sub(r'[^\w\s]', ' ', nom)
    nom = unidecode(nom)
    nom = re.sub(r'\s+', ' ', nom)
    return nom.strip()

def nettoyer_avance(nom):
    nom = nettoyer_nom(nom)
    nom = re.sub(r'b\s*p\s*\d+', '', nom)
    nom = re.sub(r'\b\d{2}\s\d{2}\s\d{2}\s\d{2}\b', '', nom)
    nom = re.sub(r'\b\d{5}\b', '', nom)
    nom = re.sub(r'\b(abidjan|ivory coast|cote d ivoire)\b', '', nom)
    nom = re.sub(r'^(ets|etablissements?|etabliessements?|group|groupe)\s+', '', nom)
    nom = re.sub(r'p\s*/\s*c\s+\w+.*$', '', nom)
    nom = re.sub(r'\s+', ' ', nom)
    return nom.strip()

def calculer_features_paire(nom1, nom2):
    lev      = fuzz.ratio(nom1, nom2) / 100
    tok_sort = fuzz.token_sort_ratio(nom1, nom2) / 100
    tok_set  = fuzz.token_set_ratio(nom1, nom2) / 100
    partial  = fuzz.partial_ratio(nom1, nom2) / 100
    diff_len = abs(len(nom1) - len(nom2)) / max(len(nom1), len(nom2), 1)
    mots1    = set(nom1.split())
    mots2    = set(nom2.split())
    total    = len(mots1.union(mots2))
    jaccard  = len(mots1.intersection(mots2)) / total if total > 0 else 0
    v1       = vectorizer.transform([nom1])
    v2       = vectorizer.transform([nom2])
    cosine   = (v1 * v2.T).toarray()[0][0]
    return [lev, tok_sort, tok_set, partial, diff_len, jaccard, cosine]

def harmoniser_nom(nom_inconnu, seuil=60):
    nom_clean         = nettoyer_avance(nom_inconnu)
    meilleur_officiel = None
    meilleure_proba   = 0
    for nom_ref in noms_officiels_list:
        score_rapide = fuzz.token_set_ratio(nom_clean, nom_ref)
        if score_rapide > seuil:
            features = calculer_features_paire(nom_clean, nom_ref)
            proba    = modele.predict_proba([features])[0][1]
            if proba > meilleure_proba:
                meilleure_proba   = proba
                mask              = noms_officiels['nom_clean'] == nom_ref
                meilleur_officiel = noms_officiels[mask]['Nom_chargeurs'].values[0]
    if meilleur_officiel is None:
        return nom_inconnu, 0.0
    return meilleur_officiel, round(meilleure_proba * 100, 2)

def niveau_confiance(confiance):
    if confiance >= 90:   return "Tres eleve"
    elif confiance >= 75: return "Eleve"
    elif confiance >= 60: return "Moyen"
    elif confiance > 0:   return "Faible"
    else:                 return "Non trouve"

def detecter_nouveaux_clients(descriptions, seuil_apparitions=5):
    noms_clean = [nettoyer_avance(n) for n in descriptions if pd.notna(n)]
    compteur   = Counter(noms_clean)
    nouveaux   = []
    for nom, count in compteur.items():
        if count >= seuil_apparitions:
            meilleur_score = max(
                [fuzz.token_set_ratio(nom, ref) for ref in noms_officiels_list[:500]],
                default=0
            )
            if meilleur_score < 60:
                nouveaux.append({
                    'Nom detecte': nom,
                    'Apparitions': count,
                    'Statut'     : 'Nouveau client potentiel'
                })
    return nouveaux

def convertir_excel(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


# ============================================================
# EN-TÊTE
# ============================================================
if logo_b64:
    logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="width:220px;height:220px;object-fit:cover;border-radius:50%;"/>'
else:
    logo_img = '<div style="font-size:80px;">⚓</div>'

st.markdown(f"""
<div class="header-zone">
    <div class="logo-ring">{logo_img}</div>
    <p class="header-eyebrow">Direction des Études Économiques · Stratégie · Planification</p>
    <h1 class="header-titre">Port Autonome d'Abidjan</h1>
    <h2 class="header-sous-titre">Système Intelligent d'Harmonisation des Chargeurs</h2>
    <p class="header-desc">
        Cet outil identifie et normalise automatiquement les noms de clients portuaires
        saisis de façon hétérogène dans le système d'information du PAA,
        grâce à un modèle de Machine Learning entraîné sur 339 902 paires de données réelles.
    </p>
    <div class="stats-bar">
        <div class="stat-item">
            <span class="stat-number">169 951</span>
            <span class="stat-label">Enregistrements</span>
        </div>
        <div class="divider-line"></div>
        <div class="stat-item">
            <span class="stat-number">98,6 %</span>
            <span class="stat-label">F1-Score</span>
        </div>
        <div class="divider-line"></div>
        <div class="stat-item">
            <span class="stat-number">62 254</span>
            <span class="stat-label">Clients officiels</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🔤  Nom par nom",
    "📂  Fichier Excel",
    "🆕  Nouveaux Clients"
])

with tab1:
    st.markdown("### Harmonisation d'un nom")
    st.markdown("---")
    nom_saisi = st.text_input(
        "Entrez le nom du client recu :",
        placeholder="Ex: 2CICS S.A, 1 2 3 DISTRIBUTION..."
    )
    if st.button("HARMONISER", key="btn_nom"):
        if not nom_saisi:
            st.warning("Veuillez entrer un nom !")
        else:
            with st.spinner("Recherche en cours..."):
                nom_harmonise, confiance = harmoniser_nom(nom_saisi)
            niveau = niveau_confiance(confiance)
            st.markdown(f"""
                <div class="resultat">
                    <h3>{nom_harmonise}</h3>
                    <p>Confiance : {confiance}% — {niveau}</p>
                </div>
            """, unsafe_allow_html=True)
            st.success(f"'{nom_saisi}' → '{nom_harmonise}' ({confiance}% — {niveau})")

with tab2:
    st.markdown("### Harmonisation d'un fichier Excel")
    st.markdown("Le fichier doit contenir une colonne **Descriptions**")
    st.markdown("---")

    if 'df_resultat' not in st.session_state:
        st.session_state.df_resultat = None
    if 'nb_lignes' not in st.session_state:
        st.session_state.nb_lignes = 50

    fichier = st.file_uploader("Chargez votre fichier Excel", type=["xlsx", "xls"])

    if fichier:
        df_charge = pd.read_excel(fichier)
        if 'Descriptions' not in df_charge.columns:
            st.error("Le fichier doit contenir une colonne 'Descriptions' !")
        else:
            total_lignes = len(df_charge)
            st.success(f"Fichier charge : {total_lignes} lignes detectees")

            st.warning("⚠️ Cette application web traite au maximum 100 lignes. Pour les gros volumes, utilisez l'application desktop.")

            st.markdown("#### Choisissez le nombre de lignes a traiter :")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("25 lignes"):
                    st.session_state.nb_lignes = min(25, total_lignes)
                    st.rerun()
            with col2:
                if st.button("50 lignes"):
                    st.session_state.nb_lignes = min(50, total_lignes)
                    st.rerun()
            with col3:
                if st.button("100 lignes"):
                    st.session_state.nb_lignes = min(100, total_lignes)
                    st.rerun()

            nb_lignes = st.number_input(
                "Ou entrez un nombre (max 100) :",
                min_value=1,
                max_value=min(100, total_lignes),
                value=min(st.session_state.nb_lignes, 100)
            )
            st.session_state.nb_lignes = nb_lignes
            st.info(f"{nb_lignes} lignes seront traitees")

            if st.button("HARMONISER LE FICHIER", key="btn_fichier"):
                df_traiter   = df_charge.head(st.session_state.nb_lignes)
                descriptions = df_traiter['Descriptions'].tolist()
                total        = len(descriptions)
                resultats    = []
                barre        = st.progress(0)
                texte        = st.empty()

                for i, nom in enumerate(descriptions):
                    nom_harmonise, confiance = harmoniser_nom(str(nom))
                    resultats.append({
                        'Description originale': nom,
                        'Nom harmonise'        : nom_harmonise,
                        'Confiance (%)'        : confiance,
                        'Niveau de confiance'  : niveau_confiance(confiance),
                    })
                    progression = int((i + 1) / total * 100)
                    barre.progress(progression)
                    texte.text(f"Traitement : {i+1}/{total} ({progression}%)")

                st.session_state.df_resultat = pd.DataFrame(resultats)
                st.success("Harmonisation terminee !")

            if st.session_state.df_resultat is not None:
                st.markdown("#### Apercu du resultat :")
                st.dataframe(st.session_state.df_resultat.head(10))
                st.download_button(
                    label="Telecharger le fichier harmonise (.xlsx)",
                    data=convertir_excel(st.session_state.df_resultat),
                    file_name="fichier_harmonise.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_btn"
                )

with tab3:
    st.markdown("### Detection des nouveaux clients")
    st.markdown("Detecte les noms qui apparaissent 5 fois ou plus mais ne sont pas dans la base officielle")
    st.markdown("---")

    fichier3 = st.file_uploader("Chargez votre fichier Excel", type=["xlsx", "xls"], key="uploader3")

    if fichier3:
        df_nouveaux = pd.read_excel(fichier3)
        if 'Descriptions' not in df_nouveaux.columns:
            st.error("Le fichier doit contenir une colonne 'Descriptions' !")
        else:
            st.success(f"Fichier charge : {len(df_nouveaux)} lignes detectees")
            if st.button("DETECTER LES NOUVEAUX CLIENTS"):
                with st.spinner("Analyse en cours..."):
                    descriptions = df_nouveaux['Descriptions'].tolist()
                    nouveaux     = detecter_nouveaux_clients(descriptions)
                if not nouveaux:
                    st.success("Aucun nouveau client detecte !")
                else:
                    st.error(f"{len(nouveaux)} nouveau(x) client(s) detecte(s) !")
                    df_nouveaux_result = pd.DataFrame(nouveaux)
                    st.dataframe(df_nouveaux_result)
                    st.download_button(
                        label="Telecharger les nouveaux clients (.xlsx)",
                        data=convertir_excel(df_nouveaux_result),
                        file_name="nouveaux_clients.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

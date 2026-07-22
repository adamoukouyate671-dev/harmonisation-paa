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

# ============================================================
# LOGO EN BASE64
# ============================================================
def get_logo_base64():
    try:
        with open("logo_paa.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

logo_b64 = get_logo_base64()
logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:160px;height:160px;object-fit:contain;"/>' if logo_b64 else '<div style="font-size:80px;">⚓</div>'

# ============================================================
# CSS GLOBAL
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap');

    .main { background-color: #0a0a0a; }
    .stApp { background-color: #0a0a0a; }
    h1, h2, h3 { color: #ffffff; }

    /* PAGE ACCUEIL */
    .accueil-wrapper {
        min-height: 90vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px 20px;
    }
    .logo-ring {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        border: 3px solid #c8a96e;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 32px auto;
        background: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #0a0a0a 100%);
        box-shadow: 0 0 40px rgba(200,169,110,0.15), 0 0 80px rgba(200,169,110,0.05);
        animation: pulse-ring 3s ease-in-out infinite;
    }
    @keyframes pulse-ring {
        0%, 100% { box-shadow: 0 0 40px rgba(200,169,110,0.15), 0 0 80px rgba(200,169,110,0.05); }
        50%       { box-shadow: 0 0 60px rgba(200,169,110,0.30), 0 0 120px rgba(200,169,110,0.10); }
    }
    .accueil-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #c8a96e;
        margin-bottom: 16px;
    }
    .accueil-titre {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(32px, 5vw, 56px);
        font-weight: 700;
        color: #ffffff;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .accueil-sous-titre {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(18px, 3vw, 28px);
        font-weight: 700;
        color: #c8a96e;
        margin-bottom: 24px;
    }
    .accueil-desc {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        font-weight: 300;
        color: #aaaaaa;
        max-width: 540px;
        line-height: 1.7;
        margin: 0 auto 48px auto;
    }
    .btn-entrer {
        display: inline-block;
        background: linear-gradient(135deg, #c8a96e 0%, #e8c97e 50%, #c8a96e 100%);
        color: #0a0a0a !important;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 18px 52px;
        border-radius: 2px;
        text-decoration: none;
        cursor: pointer;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 24px rgba(200,169,110,0.3);
    }
    .btn-entrer:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(200,169,110,0.5);
    }
    .stats-bar {
        display: flex;
        gap: 48px;
        justify-content: center;
        margin-top: 64px;
        padding-top: 40px;
        border-top: 1px solid #1f1f1f;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 32px;
        font-weight: 700;
        color: #c8a96e;
        display: block;
    }
    .stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #666666;
        margin-top: 4px;
        display: block;
    }
    .divider-line {
        width: 1px;
        height: 48px;
        background: #2a2a2a;
        margin: auto 0;
    }

    /* APP PRINCIPALE */
    .titre {
        background: linear-gradient(135deg, #1a1a1a 0%, #111111 100%);
        padding: 20px 32px;
        border-left: 4px solid #c8a96e;
        border-radius: 4px;
        text-align: center;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
    }
    .titre h1 { color: #ffffff; font-size: 22px; margin: 0; font-family: 'Inter', sans-serif; font-weight: 600; }
    .titre p  { color: #c8a96e; margin: 4px 0 0 0; font-size: 13px; letter-spacing: 1px; }
    .resultat {
        background-color: #1a1a1a;
        padding: 20px;
        border-left: 4px solid #c8a96e;
        border-radius: 4px;
        margin-top: 15px;
    }
    .resultat h3 { color: #c8a96e; margin: 0; }
    .resultat p  { color: #cccccc; margin: 5px 0 0 0; }
    .stButton button {
        background-color: #ff1a1a;
        color: white;
        border: none;
        border-radius: 3px;
        font-weight: bold;
        padding: 10px 20px;
        font-family: 'Inter', sans-serif;
    }
    .stButton button:hover { background-color: #ff4d4d; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        color: #aaaaaa;
        border-radius: 4px 4px 0 0;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background-color: #c8a96e !important;
        color: #0a0a0a !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'page' not in st.session_state:
    st.session_state.page = 'accueil'
if 'df_resultat' not in st.session_state:
    st.session_state.df_resultat = None

# ============================================================
# PAGE ACCUEIL
# ============================================================
if st.session_state.page == 'accueil':

    st.markdown(f"""
    <div class="accueil-wrapper">

        <div class="logo-ring">
            {logo_html}
        </div>

        <p class="accueil-eyebrow">Port Autonome d'Abidjan — DEESP</p>

        <h1 class="accueil-titre">Système Intelligent</h1>
        <h2 class="accueil-sous-titre">d'Harmonisation des Chargeurs</h2>

        <p class="accueil-desc">
            Un outil de Machine Learning conçu pour la Direction des Études Économiques,
            de la Stratégie et de la Planification du PAA. Il identifie et normalise
            automatiquement les noms de clients portuaires saisis de façon hétérogène.
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

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("ACCÉDER AU SYSTÈME", key="btn_entrer"):
            st.session_state.page = 'app'
            st.rerun()

# ============================================================
# PAGE APPLICATION
# ============================================================
else:
    # ── CHARGEMENT ──────────────────────────────────────────
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

    # ── FONCTIONS ────────────────────────────────────────────
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
                    nouveaux.append({'Nom detecte': nom, 'Apparitions': count, 'Statut': 'Nouveau client potentiel'})
        return nouveaux

    def convertir_excel(df):
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer

    # ── EN-TÊTE ──────────────────────────────────────────────
    col_logo, col_titre = st.columns([1, 8])
    with col_logo:
        if logo_b64:
            st.markdown(f'<img src="data:image/png;base64,{logo_b64}" style="width:60px;margin-top:8px;"/>', unsafe_allow_html=True)
    with col_titre:
        st.markdown("""
            <div class="titre">
                <div>
                    <h1>PORT AUTONOME D'ABIDJAN</h1>
                    <p>Système Intelligent d'Harmonisation des Noms Clients — DEESP</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Bouton retour accueil
    if st.button("← Accueil", key="btn_retour"):
        st.session_state.page = 'accueil'
        st.rerun()

    # ── ONGLETS ──────────────────────────────────────────────
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

        fichier = st.file_uploader("Chargez votre fichier Excel", type=["xlsx", "xls"])

        if fichier:
            df_charge = pd.read_excel(fichier)
            if 'Descriptions' not in df_charge.columns:
                st.error("Le fichier doit contenir une colonne 'Descriptions' !")
            else:
                total_lignes = len(df_charge)
                st.success(f"Fichier charge : {total_lignes} lignes detectees")
                st.markdown("#### Choisissez le nombre de lignes a traiter :")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("100 lignes"):
                        st.session_state.nb_lignes = 100
                with col2:
                    if st.button("500 lignes"):
                        st.session_state.nb_lignes = 500
                with col3:
                    if st.button("1000 lignes"):
                        st.session_state.nb_lignes = 1000
                with col4:
                    if st.button("Toutes les lignes"):
                        st.session_state.nb_lignes = total_lignes

                nb_lignes = st.number_input(
                    "Ou entrez un nombre personalise :",
                    min_value=1, max_value=total_lignes,
                    value=min(100, total_lignes)
                )
                st.info(f"{nb_lignes} lignes seront traitees")

                if st.button("HARMONISER LE FICHIER", key="btn_fichier"):
                    df_traiter   = df_charge.head(nb_lignes)
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

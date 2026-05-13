import streamlit as st
import pandas as pd
import pickle
import re
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
from collections import Counter

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Harmonisation PAA",
    page_icon="⚓",
    layout="wide"
)

# ============================================================
# STYLES CSS
# ============================================================
st.markdown("""
    <style>
    .main { background-color: #0a0a0a; }
    .stApp { background-color: #0a0a0a; }

    h1, h2, h3 { color: #ffffff; }

    .titre {
        background-color: #1a1a1a;
        padding: 20px;
        border-left: 5px solid #ff1a1a;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 20px;
    }

    .titre h1 {
        color: #ffffff;
        font-size: 28px;
        margin: 0;
    }

    .titre p {
        color: #ff1a1a;
        margin: 5px 0 0 0;
    }

    .resultat {
        background-color: #1a1a1a;
        padding: 20px;
        border-left: 5px solid #ff1a1a;
        border-radius: 5px;
        margin-top: 15px;
    }

    .resultat h3 {
        color: #ffdd00;
        margin: 0;
    }

    .resultat p {
        color: #cccccc;
        margin: 5px 0 0 0;
    }

    .stButton button {
        background-color: #ff1a1a;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        padding: 10px 20px;
    }

    .stButton button:hover {
        background-color: #ff4d4d;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        color: white;
        border-radius: 5px 5px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ff1a1a !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DU MODELE (une seule fois)
# ============================================================
@st.cache_resource
def charger_modele():
    with open("modele_random_forest.pkl", 'rb') as f:
        modele = pickle.load(f)
    return modele

@st.cache_resource
def charger_base():
    df_base = pd.read_excel("base_nettoyee.xlsx")
    vectorizer = TfidfVectorizer()
    tous_les_noms = pd.concat(
        [df_base['nom_clean'], df_base['desc_clean']]).fillna('')
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
    nom = str(nom)
    nom = nom.lower()
    nom = nom.replace("s.a.r.l.", "sarl")
    nom = nom.replace("s.a.r.l", "sarl")
    nom = nom.replace("s.a.", "sa")
    nom = nom.replace("s.a", "sa")
    nom = nom.replace("cie.", "cie")
    nom = nom.replace("gie.", "gie")
    nom = nom.replace("b.p.", "bp")
    nom = nom.replace("b.p", "bp")
    nom = re.sub(r'[^\w\s]', ' ', nom)
    nom = unidecode(nom)
    nom = re.sub(r'\s+', ' ', nom)
    nom = nom.strip()
    return nom


def calculer_features_paire(nom1, nom2):
    lev = fuzz.ratio(nom1, nom2) / 100
    tok_sort = fuzz.token_sort_ratio(nom1, nom2) / 100
    tok_set = fuzz.token_set_ratio(nom1, nom2) / 100
    partial = fuzz.partial_ratio(nom1, nom2) / 100
    diff_len = abs(len(nom1) - len(nom2)) / max(len(nom1), len(nom2), 1)
    mots1 = set(nom1.split())
    mots2 = set(nom2.split())
    total_mots = len(mots1.union(mots2))
    jaccard = len(mots1.intersection(mots2)) / total_mots if total_mots > 0 else 0
    v1 = vectorizer.transform([nom1])
    v2 = vectorizer.transform([nom2])
    cosine = (v1 * v2.T).toarray()[0][0]
    return [lev, tok_sort, tok_set, partial, diff_len, jaccard, cosine]


def harmoniser_nom(nom_inconnu, seuil=70):
    nom_clean = nettoyer_nom(nom_inconnu)
    meilleur_officiel = None
    meilleure_proba = 0

    for nom_ref in noms_officiels_list:
        score_rapide = fuzz.token_set_ratio(nom_clean, nom_ref)
        if score_rapide > seuil:
            features = calculer_features_paire(nom_clean, nom_ref)
            proba = modele.predict_proba([features])[0][1]
            if proba > meilleure_proba:
                meilleure_proba = proba
                mask = noms_officiels['nom_clean'] == nom_ref
                meilleur_officiel = noms_officiels[mask]['Nom_chargeurs'].values[0]

    if meilleur_officiel is None:
        return nom_inconnu, 0.0

    return meilleur_officiel, round(meilleure_proba * 100, 2)


def detecter_nouveaux_clients(descriptions, seuil_apparitions=5):
    noms_clean = [nettoyer_nom(n) for n in descriptions if pd.notna(n)]
    compteur = Counter(noms_clean)
    nouveaux = []
    for nom, count in compteur.items():
        if count >= seuil_apparitions:
            meilleur_score = max(
                [fuzz.token_set_ratio(nom, ref)
                 for ref in noms_officiels_list[:500]],
                default=0
            )
            if meilleur_score < 70:
                nouveaux.append({
                    'Nom détecté': nom,
                    'Apparitions': count,
                    'Statut': '🆕 Nouveau client potentiel'
                })
    return nouveaux


# ============================================================
# INTERFACE
# ============================================================

# TITRE
st.markdown("""
    <div class="titre">
        <h1>⚓ PORT AUTONOME D'ABIDJAN</h1>
        <p>Système Intelligent d'Harmonisation des Noms Clients</p>
    </div>
""", unsafe_allow_html=True)

# ONGLETS
tab1, tab2, tab3 = st.tabs([
    "🔤  Nom par nom",
    "📂  Fichier Excel",
    "🆕  Nouveaux Clients"
])


# ══════════════════════════════════════════
# ONGLET 1 : NOM PAR NOM
# ══════════════════════════════════════════
with tab1:
    st.markdown("### Harmonisation d'un nom")
    st.markdown("---")

    nom_saisi = st.text_input(
        "Entrez le nom du client reçu :",
        placeholder="Ex: 2CICS S.A, 1 2 3 DISTRIBUTION..."
    )

    if st.button("🔍 HARMONISER", key="btn_nom"):
        if not nom_saisi:
            st.warning("Veuillez entrer un nom !")
        else:
            with st.spinner("Recherche en cours..."):
                nom_harmonise, confiance = harmoniser_nom(nom_saisi)

            st.markdown(f"""
                <div class="resultat">
                    <h3>✅ {nom_harmonise}</h3>
                    <p>Confiance : {confiance}%</p>
                </div>
            """, unsafe_allow_html=True)

            st.success(f"'{nom_saisi}'  →  '{nom_harmonise}'  ({confiance}%)")


# ══════════════════════════════════════════
# ONGLET 2 : FICHIER EXCEL
# ══════════════════════════════════════════
with tab2:
    st.markdown("### Harmonisation d'un fichier Excel")
    st.markdown("Le fichier doit contenir une colonne **Descriptions**")
    st.markdown("---")

    fichier = st.file_uploader(
        "Chargez votre fichier Excel",
        type=["xlsx", "xls"]
    )

    if fichier:
        df_charge = pd.read_excel(fichier)

        if 'Descriptions' not in df_charge.columns:
            st.error("❌ Le fichier doit contenir une colonne 'Descriptions' !")
        else:
            st.success(f"✅ Fichier chargé : {len(df_charge)} lignes détectées")

            # Choix du nombre de lignes
            st.markdown("#### Choisissez le nombre de lignes à traiter :")

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
                if st.button("⚡ Toutes les lignes"):
                    st.session_state.nb_lignes = len(df_charge)

            nb_lignes = st.number_input(
                "Ou entrez un nombre personnalisé :",
                min_value=1,
                max_value=len(df_charge),
                value=min(100, len(df_charge))
            )

            st.info(f"📊 {nb_lignes} lignes seront traitées")

            if st.button("⚡ HARMONISER", key="btn_fichier"):
                df_traiter = df_charge.head(nb_lignes)
                descriptions = df_traiter['Descriptions'].tolist()
                total = len(descriptions)

                resultats = []
                barre = st.progress(0)
                texte = st.empty()

                for i, nom in enumerate(descriptions):
                    nom_harmonise, _ = harmoniser_nom(str(nom))
                    resultats.append({
                        'Descriptions': nom,
                        'Nom_harmonisé': nom_harmonise
                    })
                    progression = int((i + 1) / total * 100)
                    barre.progress(progression)
                    texte.text(f"Progression : {i+1}/{total} ({progression}%)")

                df_resultat = pd.DataFrame(resultats)

                st.success("✅ Harmonisation terminée !")
                st.markdown("#### Aperçu du résultat :")
                st.dataframe(df_resultat.head(10))

                # Télécharger
                output = df_resultat.to_excel(index=False)
                st.download_button(
                    label="📥 Télécharger le fichier harmonisé",
                    data=df_resultat.to_csv(index=False).encode('utf-8'),
                    file_name="fichier_harmonise.csv",
                    mime="text/csv"
                )


# ══════════════════════════════════════════
# ONGLET 3 : NOUVEAUX CLIENTS
# ══════════════════════════════════════════
with tab3:
    st.markdown("### Détection des nouveaux clients")
    st.markdown(
        "Détecte les noms qui apparaissent **5 fois ou plus** "
        "mais ne sont pas dans la base officielle"
    )
    st.markdown("---")

    fichier3 = st.file_uploader(
        "Chargez votre fichier Excel",
        type=["xlsx", "xls"],
        key="uploader3"
    )

    if fichier3:
        df_nouveaux = pd.read_excel(fichier3)

        if 'Descriptions' not in df_nouveaux.columns:
            st.error("❌ Le fichier doit contenir une colonne 'Descriptions' !")
        else:
            st.success(
                f"✅ Fichier chargé : {len(df_nouveaux)} lignes détectées")

            if st.button("🔍 DÉTECTER LES NOUVEAUX CLIENTS"):
                with st.spinner("Analyse en cours..."):
                    descriptions = df_nouveaux['Descriptions'].tolist()
                    nouveaux = detecter_nouveaux_clients(descriptions)

                if not nouveaux:
                    st.success("✅ Aucun nouveau client détecté !")
                else:
                    st.error(
                        f"🆕 {len(nouveaux)} nouveau(x) client(s) détecté(s) !")
                    df_nouveaux_result = pd.DataFrame(nouveaux)
                    st.dataframe(df_nouveaux_result)

                    st.download_button(
                        label="📥 Télécharger les nouveaux clients",
                        data=df_nouveaux_result.to_csv(
                            index=False).encode('utf-8'),
                        file_name="nouveaux_clients.csv",
                        mime="text/csv"
                    )
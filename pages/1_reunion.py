import locale
import shutil
import pytz
import streamlit as st
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from tools import fetch_and_concatenate_documents, load_and_save_to_faiss_json, remove_data


load_dotenv()


##################SETUP DES LOGS###################
# Ensure the logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # Create the directory if it doesn't exist


script_name = os.path.splitext(os.path.basename(__file__))[0]
log_file = os.path.join(log_dir, f"{script_name}.warn.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Écrire dans le fichier log
        logging.StreamHandler()  # Écrire dans la console (terminal)
    ]
)
##############################################

# Définir la langue en français
locale.setlocale(locale.LC_TIME, "fr_FR")

# Fuseau horaire de la Nouvelle-Calédonie
noumea_tz = pytz.timezone("Pacific/Noumea")
# Obtenir la date actuelle en Nouvelle-Calédonie
current_date_noumea = datetime.now(noumea_tz).strftime("%A %d %B %Y")

# Obtenez la date actuelle
current_date = datetime.now().strftime("%Y-%m-%d")


SYSTEM_PROMPT = f"""
Aujourd'hui, nous sommes le {current_date}({current_date_noumea}).

Tu es un assistant intelligent conçu pour aider un Etudiant à organiser ses révisions et à créer un emploi du temps adapté.
Ton rôle est d'offrir des conseils précis sur la gestion du temps. 
**Présente le planning final sous forme de tableau pour faciliter la lecture.**
"""


PROMPT_TEMPLATE_REUNION="""
Vous avez les emplois du temps de deux étudiants détaillant leurs cours et leurs horaires. Ils souhaitent planifier des réunions communes entre le {start_date} et le {end_date}.

#### **Données des étudiants :**

- **Etudiant 1 :**  
{edt1}

- **Etudiant 2 :**  
{edt2}

#### **Objectif :**
Identifier des créneaux de réunion où les deux étudiants sont disponibles en dehors de leurs cours, en respectant les contraintes suivantes :
1. **Durée des créneaux :** Chaque créneau doit **toujours** durer entre 1 heure et 2 heures .  
2. **Non-chevauchement :** Les créneaux proposés ne doivent pas empiéter sur les horaires des cours.  
3. **Période de la journée :** Les créneaux doivent être situés entre 07:00 et 18:00. Aucun créneau ne doit être proposé en dehors de cette plage horaire.  
4. **Optimisation :** Maximiser les périodes communes disponibles pour faciliter la réunion.

#### **Méthodologie :**
1. **Assomption par défaut :** Si rien dans le contexte n'est précisé concernant une horraire, considérer que l'Etudiant est entièrement disponible sur la plage horaire définie (08:00 à 18:00).  
2. **Fusion des disponibilités :** Identifier les périodes communes en croisant les plages horaires libres des deux étudiants. Par exemple, si :
   - Etudiant 1 est libre de 10h00 à 14h00
   - Etudiant 2 est libre de 11h00 à 15h00  
   Le créneau commun serait de 11h00 à 14h00.  
3. **Organisation par jour :** Proposer les créneaux , en respectant la durée minimale et maximale des réunions.  
4. **Respect des horaires :** Limiter les suggestions aux heures comprises entre 08:00 et 18:00. Ignorer toute disponibilité en dehors de cette plage horaire.  
5. **Visualisation claire :** Fournir un tableau ou une liste structurée des disponibilités communes, triées par jour.

#### **Résultat attendu :**
Un tableau clair et structuré des créneaux disponibles, organisé par jour de la semaine, qui facilite la sélection des meilleurs moments pour une réunion.
"""



def generate_planning_for_2(querry_text,main_user,second_user,list_of_dates):
    try:
        # Utilisez model_name au lieu de model
        chat_model = ChatOpenAI(model_name="gpt-4o-mini")
        logging.info(f"Modèle initialisé")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation du modèle : {repr(e)}")
        return "Erreur lors de l'initialisation du modèle."

    #On supprime parce que j'arrive pas trop à gérer les doublons for now
    st.write("Génération des fichiers ...")

    remove_data("faiss_data")
    #Création données pour les deux étudiants avec l'embeding et tout le tralala
    st.write(f"Génération pour {main_user}")
    load_and_save_to_faiss_json(main_user)
    st.write(f"Génération pour {second_user}")
    load_and_save_to_faiss_json(second_user)

    st.write("Création du contexte ...")
    #On récupère les données et on les manipules pour récupérer tout en une chaine de caractere 
    context_main_user=fetch_and_concatenate_documents(querry_text,list_of_dates,main_user)
    context_second_user=fetch_and_concatenate_documents(querry_text,list_of_dates,second_user)
    
    # Crée le prompt à partir du template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_REUNION)
    prompt = prompt_template.format(edt1=context_main_user,edt2=context_second_user,start_date=list_of_dates[0],end_date=list_of_dates[len(list_of_dates)-1])
    
    # Ajoute le message système pour guider le comportement de l'IA
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    logging.info(f"Message envoyé à l'IA:\n {messages}")
    # Utilise le modèle de chat pour générer la réponse
    st.write("Génération de la réponse")
    try:
        st.write(messages[1])
        response = chat_model(messages=messages)
        status.update(label="Download complete!", state="complete", expanded=False )
        return response.content  # Renvoie le contenu de la réponse générée
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse: {repr(e)}")
        return "Erreur lors de la génération de la réponse."

st.set_page_config(page_title="Réunion", page_icon="📅")

st.markdown("# Réunion")
st.write(
    """Création de réunion avec d'autres étudiants"""
)



user_id1 = st.text_input("Entrez votre identifiant: ", "rcastelain")
user_id2 = st.text_input("Entrez deuxième identifiant: ", "htiaiba")
# Sélection des dates
date_debut = st.date_input("Sélectionner la date de début pour la réunion :")
#Pour l'instant on met pas vu qu'on a plus de cours
#date_debut = st.date_input("Sélectionner la date de début pour la réunion :", min_value=date.today())
# Calcul de la date de fin max (14 jours après la date de début)
date_fin_max = date_debut + timedelta(weeks=2)

date_fin = st.date_input("Sélectionner la date de fin :", min_value=date_debut, max_value=date_fin_max)

# Générer une liste de dates entre date_debut et date_fin pour les utiliser dans la recherche
if date_fin >= date_debut:
    liste_dates = [
        (date_debut + timedelta(days=i)).isoformat()
        for i in range((date_fin - date_debut).days + 1)
    ]
    
# Bouton pour valider les entrées
if st.button("Création du planning"):
    # Appel de la fonction avec les entrées
    with st.status("Génération de la réponse...", expanded=True) as status:
        resultat = generate_planning_for_2(f"Cours entre date_debut : {date_debut}  date_fin : {date_fin}",user_id1, user_id2, liste_dates)

    # Affichage du résultat
    st.write(resultat)



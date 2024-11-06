from langchain_openai import ChatOpenAI
from datetime import datetime
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from streamlit_calendar import calendar
from scrap_edt import get_edt,get_edt_semaine
from faiss_handler import transform_to_documents,save_to_faiss,retrieve_documents,transform_weeks_to_documents
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
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

# Obtenez la date actuelle
current_date = datetime.now().strftime("%Y-%m-%d")


# System Prompt to define the assistant's role
SYSTEM_PROMPT = f"""
Aujourd'hui, nous sommes le {current_date}.

Tu es un assistant intelligent conçu pour aider un étudiant à organiser ses révisions et à créer un emploi du temps adapté. 
Ton rôle est d'offrir des conseils précis sur la gestion du temps, la répartition des matières, et les stratégies de révision efficaces. 
Tu dois poser des questions pour bien comprendre les objectifs de l'étudiant, ses priorités, et ses échéances. 
Tu es là pour l'accompagner dans ses révisions en proposant des suggestions d'amélioration et en lui fournissant des explications claires et adaptées à ses besoins.
"""

# Chat prompt template
PROMPT_TEMPLATE = """
Réponds à la question suivante en utilisant le contexte ci-dessous. Vérifie que les informations utilisées concordent avec les questions de l'humain. Si tu n'es pas sûr de la réponse, n'hésite pas à demander des précisions.

{context}

---

Réponds à la question en utilisant le contexte ci-dessus : {question}

Exemple :
Question : Quel est le début de mon cours de Mathématiques ?
Réponse : Le début de votre cours de Mathématiques est le 10 janvier.
"""
    
def save_user_edt_to_faiss(user_id):
    data=get_edt_semaine(user_id)
    docs=transform_weeks_to_documents(data,user_id)
    save_to_faiss(docs)

def generate_response(querry_text, user_id):
    
    try:
        # Utilisez model_name au lieu de model
        chat_model = ChatOpenAI(model_name="gpt-4")
        logging.info(f"Modèle initialisé")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation du modèle : {repr(e)}")
        return "Erreur lors de l'initialisation du modèle."

    # Récupère les documents pertinents à partir de FAISS
    results = retrieve_documents(querry_text, user_id, 30)
    
    # Concatène tous les documents récupérés pour former le contexte
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    # Crée le prompt à partir du template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=querry_text)
    
    # Ajoute le message système pour guider le comportement de l'IA
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    # Utilise le modèle de chat pour générer la réponse
    try:
        response = chat_model(messages=messages)
        return response.content  # Renvoie le contenu de la réponse générée
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse: {repr(e)}")
        return "Erreur lors de la génération de la réponse."
    
# Fonction principale pour la page web
def main():
        
    st.title("Chatbot🤖")

    user_id = st.text_input("Entrez votre identifiant: ", "")

    if 'edt' not in st.session_state:
        st.session_state.edt = None

    if st.button("Valider"):
        if user_id:
            try:
                cours = get_edt(user_id)
                st.session_state.edt = cours
                st.success("Identifiant validé ! Voici votre emploi du temps :")
                
                #Envoie et embeding des informations de l'utilisateur
                data=get_edt(user_id)
                docs=transform_to_documents(data,user_id)
                save_to_faiss(docs)
                
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Veuillez entrer un identifiant valide.")

    # Définition des événements avec les couleurs pour TD, TP, CM
    if st.session_state.edt:
        events = []
        for cours in st.session_state.edt:
            
            color = ""
            if "TD" and "Td" in cours['nom_cours']:
                color = "#32a852"  # TD en vert clair
            elif "TP" and "Tp" in cours['nom_cours']:
                color = "#ffb22e"  # TP en orange clair
            elif "CM" and "Cm" in cours['nom_cours']:
                color = "#4287f5"  # CM en bleu clair
            else:
                color = "gray"  # Sinon par défaut en gris

            events.append({
                'start': cours['début'],
                'end': cours['fin'],
                'title': cours['nom_cours'],
                'description': cours['description'],
                'color': color
            })
    else:
        events = []

    # Affichage du calendrier dans un conteneur
    calendar(
    events=events,
    options={
        "locale": "fr",  # Tout en français
        'initialView': 'dayGridMonth',  # Vue par défaut : mois
        'editable': True,
        'headerToolbar': {
            'left': 'prev,today,next',
            'center': 'title',
            'right': 'dayGridMonth,timeGridWeek,timeGridDay'
        },
        "buttonText": {
            "today": "Aujourd'hui",  # Change le texte du bouton Today
            "month": "Mois",  # Change le texte du mode Mois
            "week": "Semaine",  # Change le texte du mode Semaine
            "day": "Jour"  # Change le texte du mode Jour
        },
        "slotMinTime": "07:00:00", #C'est l'heure minimale sur le calendrier
        "slotMaxTime": "19:00:00", #C'est l'heure maximale sur le calendrier
        "scrollTime": "07:00:00", # C'est l'heure à laquelle le calendrier en mode jour ou semaine commence
        "allDaySlot": False,  # permet ici de désactiver l'affichage des événements "All Day" (prennait de la place)
        "height": 'auto',  # La hauteur s'ajustera automatiquement ici *-*
        "contentHeight": 'auto',  # ça évite le défilement vertical dans le calendrier
        "expandRows": True,  # Prends/remplit l'espace disponible
        "eventMaxHeight": 20,  # Hauteur maximale des événements
        "stickyHeaderDates": True, # Permet de garder l'en-tête fixe, si l'utilisateur doit défiler son emploi du temps pendant le mode semaine/jour

    },
    custom_css="""
    .fc-event-past { opacity: 0.8; } #Change l'opacité des événements déjà passé
    .fc-event-time { font-style: italic; }
    .fc-event-title { font-weight: 700; font-size: 0.85rem; }  # Réduire la taille du texte des événements
    .fc-daygrid-event { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }  # Limiter la largeur des événements et ajouter des points de suspension
    .fc-daygrid-block-event { max-height: 20px; }  # Restreindre la hauteur des événements dans la vue "Mois"
    .fc-toolbar-title { font-size: 2rem; }
    .fc-timegrid-slot { height: auto !important; }  # Ajuste la hauteur des lignes dans la vue "semaine" et "jour"
    .fc-day-today {background: #f5f5f5 !important;}
    """
    )

    # Interaction avec le chatbot
    user_input = st.text_input("Vous: ", "")
    if st.button("Envoyer"):
        if user_input:
            st.write(f"Vous: {user_input}")
            bot_response = generate_response(user_input,user_id)
            st.write(f"Chatbot: {bot_response}")
        else:
            st.write("Veuillez entrer un message.")

def test():
    user_id="rcastelain"
    data=get_edt(user_id)
    docs=transform_to_documents(data,user_id)
    save_to_faiss(docs)
    results=retrieve_documents("Quels sont les cours pour la semaine du 21 au 27 octobre",user_id)
    print(results)

if __name__ == "__main__":
    main()

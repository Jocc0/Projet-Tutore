import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
from ics import Calendar
import pytz
from streamlit_calendar import calendar

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# Fonction pour récupérer l'EDT depuis l'URL ICS
def get_edt(user_id):
    ics_url = f"http://applis.univ-nc.nc/cgi-bin/WebObjects/EdtWeb.woa/2/wa/default?login={user_id}%2Fical"
    response = requests.get(ics_url)

    # S'assurer que ça soit bien encoder en UTF-8
    response.encoding = "UTF-8"
    
    if response.ok:
        ics_content = response.text
    else:
        raise Exception("Veuillez entrer un identifiant valide 🚫")

    cal = Calendar(ics_content)
    cours = []

    local_tz = pytz.timezone("Pacific/Noumea")

    for event in cal.events:
        start_local = event.begin.astimezone(local_tz).strftime('%Y-%m-%d %H:%M')
        end_local = event.end.astimezone(local_tz).strftime('%Y-%m-%d %H:%M')

        cours.append({
            "nom_cours": event.name,
            "début": start_local,
            "fin": end_local,
            "description": event.description
        })

    return cours

# Fonction pour générer une réponse via OpenAI
def generate_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content

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
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Veuillez entrer un identifiant valide.")

    # Pour choisir le mode de calendrier
    mode = st.selectbox("Choisissez le mode de calendrier:", ["dayGridMonth", "timeGridWeek", "timeGridDay"])

    # Définition des spécificités ou événement récupérer pour le calendrier
    if st.session_state.edt:
        events = [
            {
                'start': c['début'],
                'end': c['fin'],
                'title': c['nom_cours'],
                'description': c['description']
            } for c in st.session_state.edt
        ]
    else:
        events = []

    # Affichage du calendrier avec CSS personnalisée
    calendar(
        events=events,
        options={
            "locale": "fr",
            'initialView': mode,
            'editable': False,
            'headerToolbar': {
                'left': 'prev,next today',
                'center': 'title',
                'right': 'dayGridMonth,timeGridWeek,timeGridDay'
            },
            "slotMinTime": "07:00:00",
            "slotMaxTime": "18:00:00",
        },
        
        custom_css="""
        .fc-event-past { opacity: 0.8; }
        .fc-event-time { font-style: italic; }
        .fc-event-title { font-weight: 700; }
        .fc-toolbar-title { font-size: 2rem; }
        .stApp { font-family: sans-serif; }
        """
    )

    # Interaction avec le chatbot
    user_input = st.text_input("Vous: ", "")
    if st.button("Envoyer"):
        if user_input:
            st.write(f"Vous: {user_input}")
            bot_response = generate_response(user_input)
            st.write(f"Chatbot: {bot_response}")
        else:
            st.write("Veuillez entrer un message.")

if __name__ == "__main__":
    main()

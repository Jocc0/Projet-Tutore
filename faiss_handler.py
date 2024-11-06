from dotenv import load_dotenv
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

#!Utiliser ça pour taleur
from langchain.text_splitter import RecursiveJsonSplitter
import logging
import os


###################PATH VARIABLES###################
DATA_PATH = "data/"
FAISS_PATH = "faiss_data"
JSON_PATH ="json_schedules"
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

# Chargement des variables d'environnement
try:
    load_dotenv()
    logging.info("Variables d'environnement chargées")
except Exception as e:
    logging.error(f"Erreur dans le chargement des variables d'environnement: {repr(e)}")

# Configuration de l'API Key
try:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
    if not os.environ["OPENAI_API_KEY"]:
        raise ValueError("OPENAI_API_KEY non défini dans les variables d'environnement.")
except Exception as e:
    logging.error(f"Erreur lors de la définition de la clé API OpenAI: {repr(e)}")

# Initialisation des embeddings
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    logging.info("Modèle initialisé")
except Exception as e:
    logging.error(f"Erreur dans l'initialisation du modèle: {repr(e)}")


def json_to_documents(cours_data,user_id):
    emploi_du_temps = json.load(cours_data)["emploi_du_temps"]


def transform_to_documents(cours_data, user_id):
    """
    Cette fonction va transformer les données de cours (retourné par get_edt)
    en un Document du format Langchain qui pourra être utilisé par FAISS.
    
    Args:
        cours_data (List): de la forme :
        [{
            "nom_cours": nom_coupee,
            "début": start_local,
            "fin": end_local,
            "description": description_coupee
        },...]
        
        user_id (str): l'id de la personne à qui appartiennent ces données

    Returns:
        Document: un document au format Langchain
    """
    documents = []  # Liste pour mettre tous les cours
    content = ""

    try:
        logging.info(f"Traitement de {len(cours_data)} cours.")
        for cours in cours_data:
            content = (
                f"Nom du cours: {cours['nom_cours']}\n"
                f"Début du cours: {cours['début']}\n"
                f"Fin du cours: {cours['fin']}\n"
                f"Description du cours: {cours['description']}\n\n"
            )
            documents.append(Document(page_content=content, metadata={"user_id": user_id,
                                                                     "source": f"http://applis.univ-nc.nc/cgi-bin/WebObjects/EdtWeb.woa/2/wa/default?login={user_id}%2Fical"}))
        return documents
    except Exception as e:
        logging.error(f"Erreur lors de la transformation des données en documents : {repr(e)}")
        return []  # Retourner une liste vide en cas d'erreur
    
def load_faiss_vector_store():
    """
    Charge le vector store FAISS à partir du chemin spécifié.
    
    Returns:
        FAISS: L'instance du vector store FAISS chargée.
    """
    if os.path.exists(FAISS_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            logging.info(f"Index FAISS local chargé depuis : {FAISS_PATH}")
            return vector_store
        except Exception as e:
            logging.error(f"Erreur lors du chargement de l'index FAISS : {repr(e)}")
            return None
    else:
        logging.info(f"Aucun index FAISS trouvé à {FAISS_PATH}")
        return None

def retrieve_documents(querry_text, user_id, top_k=1):
    vector_store = load_faiss_vector_store()  # Charger le vector store ici
    if vector_store:
        try:
            # On recherche dans le vector store FAISS
            results = vector_store.similarity_search(
                querry_text, k=top_k, filter={"user_id": user_id}
            )
            if results is not None:
                logging.info(f"{len(results)}informations intéressante trouvée pour {user_id}")
                return results
            else:
                logging.info("Aucune information intéressante trouvée")
        except Exception as e:
            logging.info(f"Erreur lors de la recherche de données : {repr(e)}")
    return None

def retrive_documents_score(querry_text,user_id,top_k=3):
    vector_store = load_faiss_vector_store()  # Charger le vector store ici
    if vector_store:
        try:
            # On recherche dans le vector store FAISS
            results = vector_store.similarity_search_with_relevance_scores(
                querry_text, k=top_k, filter={"user_id": user_id}
            )
            if results is not None:
                logging.info(f"{len(results)}informations intéressante trouvée pour {user_id}")
                return results
            else:
                logging.info("Aucune information intéressante trouvée")
        except Exception as e:
            logging.info(f"Erreur lors de la recherche de données : {repr(e)}")
    return None




def save_to_faiss(documents: list[Document]):
    """
    Sauvegarde un document à un vector store de FAISS.

    Args:
        documents (list[Document]): une liste de documents au format Langchain
    """
    try:
        vector_store = load_faiss_vector_store()
        # Création du vector store
        if not vector_store:
            logging.info("Création du vector store FAISS.")
            # Si le fichier n'existe pas, on le crée
            # On s'assure de toujours respecter les dimensions des vecteurs du modèle
            index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
            logging.info("Création du vector store FAISS.")
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
    except Exception as e:
        logging.error(f"Erreur lors de la création de l'index FAISS : {repr(e)}")
    try:
        vector_store.add_documents(documents=documents)
        vector_store.save_local(FAISS_PATH)
        logging.info(f"Sauvegardé {len(documents)} documents dans {FAISS_PATH}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des documents dans FAISS : {repr(e)}")



def transform_weeks_to_documents(cours_data, user_id):
    """
    Cette fonction va transformer les données de cours (retourné par get_edt)
    en un Document du format Langchain qui pourra être utilisé par FAISS.
    
    Args:
        cours_data (List): de la forme :
        [{
            "nom_cours": nom_coupee,
            "début": start_local,
            "fin": end_local,
            "description": description_coupee
        },...]
        
        user_id (str): l'id de la personne à qui appartiennent ces données

    Returns:
        Document: un document au format Langchain
    """
    documents = []  # Liste pour mettre tous les cours
    content = ""

    try:
        logging.info(f"Traitement des cours.")
        for semaine in cours_data:
            content=""
            for cours in semaine:
                content += (
                    f"Nom du cours: {cours['nom_cours']}\n"
                    f"Début du cours: {cours['début']}\n"
                    f"Fin du cours: {cours['fin']}\n"
                    f"Description du cours: {cours['description']}\n\n"
                )
            documents.append(Document(page_content=content, metadata={"user_id": user_id,"source": f"http://applis.univ-nc.nc/cgi-bin/WebObjects/EdtWeb.woa/2/wa/default?login={user_id}%2Fical"}))
            logging.info(f"Documents de {len(content)}caractères créés pour l'utilisateur {user_id}")
        return documents
    except Exception as e:
        logging.error(f"Erreur lors de la transformation des données en documents : {repr(e)}")
        return []  # Retourner une liste vide en cas d'erreur
    

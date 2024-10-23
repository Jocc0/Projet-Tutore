from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_KEY")
        
DATA_PATH= "data/"
CHROMA_PATH = "chroma"

def transform_to_documents(cours_data,user_id):
    """_summary_
    Cette fonction va transformer les données de cours (retourné par get_edt) en un Document du format Langchain
    qui pourra être utilisé par chroma
    
    Args:
        cours_data (List): les données d'une liste celle de get_edt 
        user_id (str): l'id de la personne à qui appartiennent ces données

    Returns:
        Document: un document format Lanchain
    """
    
    documents=[] #Liste pour mettre tous les cours
    content=""
    #Boucle sur chaque cours et on l'ajoute au contenu du fichier

    print(len(cours_data))
    for cours in cours_data:
        print(len(cours_data))
        # print(f"\n\n\n\n\n{cours}\n\n\n\n\n")
        content = (
                f"Nom du cours: {cours['nom_cours']}\n"
                f"Début: {cours['début']}\n"
                f"Fin: {cours['fin']}\n"
                f"Description: {cours['description']}\n\n"
            )
        documents.append(Document(page_content=content,metadata={"user_id":user_id,
                                                    "source": f"http://applis.univ-nc.nc/cgi-bin/WebObjects/EdtWeb.woa/2/wa/default?login={user_id}%2Fical"
                                                    }))
    #Création du document LangChain avec le contenu des cours
    return documents
    
def save_to_chroma(documents: list[Document],collection_name="default_collection"):
    """_summary_
        Sauvegarde un document à un vector store de Chroman, organisé dans une collection 
        (La fonction du collection, c'est pour le futur si jamais je veux ajouter des trucs style 
        des documents d'actu, par exemple des articles de l'université ou bien des fichiers de l'utilisateur
        j'aimerais que les données soient rangées en collection))
    Args:
        document (Document): un document format Lanchain
        collection_name (str): le nom de la collection. Defaults to "default_collection".
    """

    # Initialisation du Chroma vector store avec les embeddings OpenAI, création ou chargement de la collection
    vector_store = Chroma(
        collection_name=collection_name, 
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    # Ajout des documents à la collection
    vector_store.add_documents(documents=documents)  # Wrap in a list, even if it's a single document

    print(f"Saved {len(documents)} documents to {CHROMA_PATH} in the collection : {collection_name}")


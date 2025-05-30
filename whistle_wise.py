from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

import os
import bs4

from urllib.request import Request, urlopen
from urllib.parse import urlparse
import ssl

import requests
from bs4 import BeautifulSoup

def get_wikipedia_glossary(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    return soup

def extract_glossary(soup, sport_name):
    terms = []
    current_term = None
    for dl in soup.find_all("dl"):
        for tag in dl.children:
            if tag.name == "dt":
                current_term = tag.text.strip()
            elif tag.name == "dd" and current_term:
                definition = tag.text.strip()
                terms.append({
                    "sport": sport_name,
                    "term": current_term,
                    "definition": definition
                })
                current_term = None
    return terms


def build_chain():
    from langchain.chains import create_retrieval_chain
    from langchain_community.document_loaders import WebBaseLoader
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDoOv96fCayBc-5vA_tBoIAFyNglFDizAQ"

    sport_glossary_urls = {
        "Basketball": "https://en.wikipedia.org/wiki/Glossary_of_basketball_terms",
        "Football": "https://en.wikipedia.org/wiki/Glossary_of_American_football",
        "Baseball": "https://en.wikipedia.org/wiki/Glossary_of_baseball_terms",
        "Hockey": "https://en.wikipedia.org/wiki/Glossary_of_ice_hockey_terms",
        "Soccer": "https://en.wikipedia.org/wiki/Glossary_of_association_football_terms",
        "Golf": "https://en.wikipedia.org/wiki/Glossary_of_golf",
        "Tennis": "https://en.wikipedia.org/wiki/Glossary_of_tennis_terms"
    }

    all_terms = []

    from langchain_core.documents import Document

    for sport, url in sport_glossary_urls.items():
        print(f"Scraping {sport} glossary...")
        soup = get_wikipedia_glossary(url)
        terms = extract_glossary(soup, sport)
        all_terms.extend(terms)


    print(f"\n✅ Total glossary terms collected: {len(all_terms)}")
    print("Sample:", all_terms[:3])


    docs = []
    for item in all_terms:
        content = f"{item['term']}: {item['definition']}"
        metadata = {"sport": item['sport'], "term": item['term']}
        docs.append(Document(page_content=content, metadata=metadata))


    # 1. Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 2. Use Gemini-based embeddings
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Gemini embedding model

    # 3. Store in vector DB
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

    # 4. Create retriever
    retriever = vectorstore.as_retriever()

    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
        "You are a kind, considerate sports terminology assistant with expert knowledge across multiple sports, including football, basketball, baseball, ice hockey, soccer, golf, and tennis. "
        "Your job is to accurately answer questions about sports terms, rules, player roles, and statistics based on the provided context. "
        "Use the retrieved context below to generate a clear and concise answer (no more than three sentences). Find the MOST relevant information for the question; the most"
        "important thing is to be concise and make sure to answer exactly what is being asked. Also I want you to specify exactly what SPORT the question is from. For example,"
        "say if the answer is for golf, football, or some other sport "
        "If the information is not found in the context, say you don't know. If the question is unclear, ask a clarifying follow-up."
        "\n\n"
        "{context}"
    )

    # the variable context is used by create_stuff_documents_chain to "stuff"/concatenate all context docs

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),  # input is a variable, also context in system prompt
        ]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    from langchain.chains.combine_documents import create_stuff_documents_chain

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # If there is no chat_history, then the input is just passed directly to the
    # retriever. If there is chat_history, then the prompt and LLM will be used to
    # generate a search query. That search query is then passed to the retriever.

    # This chain prepends a rephrasing of the input query to our retriever,
    # so that the retrieval incorporates the context of the conversation.

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # history_aware_retriever and question_answer_chain in sequence, retaining
    # intermediate outputs such as the retrieved context for convenience.
    # It has input keys input and chat_history, and includes input, chat_history,
    # context, and answer in its output.
    rag_chain = create_retrieval_chain(history_aware_retriever,
                                    question_answer_chain)

    # compare code above to QA chain (from the previos section)
    #question_answer_chain = create_stuff_documents_chain(llm, prompt)
    #rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain





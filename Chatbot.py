from openai import OpenAI
import streamlit as st
from streamlit.logger import get_logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.graphs import Neo4jGraph
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OllamaEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import List, Any
from langchain.chains import GraphCypherQAChain
from langchain.vectorstores.neo4j_vector import Neo4jVector


#my shit
llm = None


def configure_qa_structure_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response based on vector search and retrieval of structured chunks

    sample_query = """
    // 0 - prepare question and its embedding
        MATCH (ch:Chunk) -[:HAS_EMBEDDING]-> (chemb)
        WHERE ch.block_idx = 19
        WITH ch.sentences AS question, chemb.value AS qemb
        // 1 - search chunk vectors
        CALL db.index.vector.queryNodes($index_name, $k, qemb) YIELD node, score
        // 2 - retrieve connectd chunks, sections and documents
        WITH node AS answerEmb, score
        MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (s:Section)
        WITH s, score LIMIT 1
        MATCH (d:Document) <-[*]- (s) <-[:HAS_PARENT*]- (chunk:Chunk)
        WITH d, s, chunk, score ORDER BY chunk.block_idx ASC
        // 3 - prepare results
        WITH d, collect(chunk) AS chunks, score
        RETURN {source: d.url, page: chunks[0].page_idx} AS metadata,
            reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, score;
    """

    general_system_template = """
    You are a customer service agent that helps a customer with answering questions about a service.
    Use the following context to answer the question at the end.
    Make sure not to make any changes to the context if possible when prepare answers so as to provide accuate responses.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    At the end of each answer you should contain metadata for relevant document in the form of (source, page).
    For example, if context has `metadata`:(source:'docu_url', page:1), you should display ('doc_url',  1).
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database='neo4j',  # neo4j by default
        index_name="chunkVectorIndex",  # vector by default
        node_label="Embedding",  # embedding node label
        embedding_node_property="value",  # embedding value property
        text_node_property="sentences",  # text by default
        retrieval_query="""
            WITH node AS answerEmb, score
            ORDER BY score DESC LIMIT 10
            MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (s:Section)
            WITH s, answer, score
            MATCH (d:Document) <-[*]- (s) <-[:HAS_PARENT*]- (chunk:Chunk)
            WITH d, s, answer, chunk, score ORDER BY d.url_hash, s.title, chunk.block_idx ASC
            // 3 - prepare results
            WITH d, s, collect(answer) AS answers, collect(chunk) AS chunks, max(score) AS maxScore
            RETURN {source: d.url, page: chunks[0].page_idx+1, matched_chunk_id: id(answers[0])} AS metadata,
                reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, maxScore AS score LIMIT 3;
    """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 25}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=7000,      # gpt-4
    )
    return kg_qa

NEO4J_URL = "neo4j+s://5d7f6789.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "MbOWv0Ivgwd14w5n8cuB7H1dydBXYXVg6CNj_UVmsek"
NEO4J_DATABASE = "neo4j"


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True, openai_api_key=openai_api_key)
    store = Neo4jVector.from_existing_index(
            embedding = OpenAIEmbeddings(openai_api_key=openai_api_key),
            url = NEO4J_URL,
            username = NEO4J_USER,
            password = NEO4J_PASSWORD,
            database='neo4j',  # neo4j by default
            index_name="chunkVectorIndex",  # vector by default
            node_label="Embedding",  # embedding node label
            embedding_node_property="value",  # embedding value property
            text_node_property="sentences",  # text by default
            retrieval_query="""
                WITH node AS answerEmb, score 
                ORDER BY score DESC LIMIT 10
                MATCH (answerEmb) <-[:HAS_EMBEDDING]- (answer) -[:HAS_PARENT*]-> (s:Section)
                WITH s, answer, score
                MATCH (d:Document) <-[*]- (s) <-[:HAS_PARENT*]- (chunk:Chunk)
                WITH d, s, answer, chunk, score ORDER BY d.url_hash, s.title, chunk.block_idx ASC
                // 3 - prepare results
                WITH d, s, collect(answer) AS answers, collect(chunk) AS chunks, max(score) AS maxScore
                RETURN {source: d.url, page: chunks[0].page_idx+1, matched_chunk_id: id(answers[0])} AS metadata, 
                    reduce(text = "", x IN chunks | text + x.sentences + '.') AS text, maxScore AS score LIMIT 3;
    """)
    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

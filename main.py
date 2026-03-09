import streamlit as st
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

st.set_page_config(page_title="AI News Analyst", layout="wide")
st.title("🤖 პირადი AI ანალიტიკოსი")
st.markdown("---")

st.sidebar.header("🔍 ძებნის პარამეტრები")
search_topic = st.sidebar.text_input("შეიყვანე სასურველი თემა:", "Samsung AI")
update_button = st.sidebar.button("სიახლეების განახლება 🔄")


def process_news(topic):
    with st.spinner(f"ვეძებ უახლეს ამბებს თემაზე: {topic}..."):
        data = newsapi.get_everything(q=topic, language='en', sort_by='publishedAt', page_size=5)
        articles = data.get('articles', [])

        if not articles:
            return None

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        docs = [
            Document(page_content=f"სათაური: {a['title']}\nაღწერა: {a.get('description', 'აღწერა არ არის')}")
            for a in articles
        ]


        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )
        return vector_db


if update_button or "vector_db" not in st.session_state:
    st.session_state.vector_db = process_news(search_topic)

if st.session_state.vector_db:
    st.sidebar.success(f"✅ სიახლეები თემაზე '{search_topic}' მზად არის!")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    template = """შენ ხარ პროფესიონალი AI ანალიტიკოსი. 

        პასუხის გაცემის წესები:
        1. პრიორიტეტი მიანიჭე მოცემულ "კონტექსტს" (სტატიებს) და პასუხი ააგე მათზე დაყრდნობით.
        2. თუ კითხვაზე პასუხი სტატიებში არ მოიძებნება, გამოიყენე შენი ზოგადი ცოდნა, თუმცა პასუხის ბოლოს აუცილებლად მიაწერე მოკლე შენიშვნა: "(წყარო: ზოგადი AI ცოდნა)".
        3. თუ ინფორმაცია საერთოდ არ მოგეპოვება, პირდაპირ თქვი, რომ ამ ეტაპზე მონაცემები არ გაქვს.

        კონტექსტი:
        {context}

        კითხვა: {question}
        პასუხი ქართულად:"""

    prompt = ChatPromptTemplate.from_template(template)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
            {"context": st.session_state.vector_db.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("ჰკითხე AI-ს ამ სიახლეების შესახებ..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            try:
                response = rag_chain.invoke(prompt_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"შეცდომა პასუხისას: {e}")
else:
    st.warning("⚠️ სიახლეები ვერ მოიძებნა. სცადე სხვა თემა.")
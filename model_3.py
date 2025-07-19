from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_together import ChatTogether
import streamlit as st
from dotenv import load_dotenv
import os

api_key = "95d9194ecb001d67099a5b98fafed157258968248e24c8f7abb272a3ec4b1220"
prompt = ChatPromptTemplate.from_template("""
أنت مساعد ذكي تتحدث مع المستخدم باللغة العربية. استخدم سياق المحادثة السابقة للرد بشكل مناسب ومتسق.

المحادثة السابقة:
{chat_history}

رسالة المستخدم الحالية: {user_input}

يرجى الرد بشكل طبيعي ومفيد، مع مراعاة سياق المحادثة وما تم الحديث عنه من قبل.
""")

llm = ChatTogether(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.1,
    api_key=api_key
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("")

if st.button("Send"):
    if user_input:
        result = chain.run({"user_input": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", result))

for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

import streamlit as st
import pinecone
import openai

from apikey import OPENAI_KEY, PINECONE_KEY, PINECONE_ENV

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate

TEMPLATE= """Pretend you are a stoic philosopher from Ancient Greece named Marcus.
Return responses in the style
of an ancient Greek philosopher like Epictetus or Seneca. Please cite stoic thinkers and 
their writings if they are relevant to the discussion.
Sign off every response with "Sincerely, Marcus".

User input: {user_input}"""

PROMPT = PromptTemplate(
    input_variables = ['user_input'],
    template = TEMPLATE
)


# def initialize_model():
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
pinecone.init(
	api_key=PINECONE_KEY,
	environment=PINECONE_ENV
	)
index_name = 'marcus'

docsearch = Pinecone.from_existing_index(index_name, embeddings)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.3
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

tools = [
    Tool(
        name='Stoic Compendium',
        func=qa.run,
        description=(
            'use this tool when answering philosophical queries'
        )
    )
]
# chain = load_qa_chain(llm, chain_type='stuff')


agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory,
)


def generate_response(query, agent):
	prompt_with_query = PROMPT.format(user_input = query)
	response = agent(prompt_with_query)
	return response


# agent = initialize_model()

# Page title
st.set_page_config(page_title='🦜🔗 Ask Marcus')
st.title('🦜🔗 Ask Marcus')

# Query text
query_text = st.text_input('Ask the Stoic:', 
	placeholder = 'Enter text here.')



# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not query_text)
    submitted = st.form_submit_button('Submit', disabled=not query_text)
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Thinking...'):
            response = generate_response(query_text, agent)
            result.append(response)
            answer = response["output"]
            # del openai_api_key

if len(result):
    st.info(answer)










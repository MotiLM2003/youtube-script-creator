import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ["OPENAI_API_KEY"] = apikey
st.title('🦜🔗 Youtube GPT')
prompt = st.text_input('Enter a youtube title subject')

title_template = PromptTemplate(
    input_variables = ['topic'],
    template= 'write me 1 creative youtube video title idea about {topic}'
)


script_template = PromptTemplate(
    input_variables = ['title'],
    template= 'write me a 3000 words youtube script this is title TITLE: {title}'
)

llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title")
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script")
sc =  SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
output_variables=['title', 'script'], verbose=True)

if prompt:
    response =sc({'topic' : prompt})
    st.write(response['title'])
    st.write(response['script'])

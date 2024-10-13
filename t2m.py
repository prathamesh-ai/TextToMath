import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text To Math Problem Solver And Data Serach Assistant")
st.title("Text To Math Problem Solver Uing Google Gemma 2")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")
if not groq_api_key:
    st.info("Please add your Groq APPI key to continue")
    st.stop()

llm=ChatGroq(model="Gemma-7b-It",groq_api_key=groq_api_key)
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"

)

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

prompt="""
You are an agent tasked for solving user's mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain = LLMChain(llm=llm,prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

assistant_agent = initialize_agent(
    tools = [wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error = True 
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant", "content":"Hi I am a Math Chatbot who can answer your maths queries."}
    ]
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    def generate_response(question):
        response = assistant_agent.invoke({'input':question})
        return response
question = st.text_area("Enter your question: ", "I have 5 bananas. I gave 1 banana to my brother, 2 bananas to my Father, and I ate 1 banana. How many bananas am I left with?")
if st.button("Get Answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                         )
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")
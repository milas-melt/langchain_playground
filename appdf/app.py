import gradio as gr
from gradio.components import File, Textbox
from langchain import OpenAI, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chains import AnalyzeDocumentChain
from dotenv import load_dotenv
load_dotenv('.env.local')

llm = OpenAI(temperature=0)
text_splitter = CharacterTextSplitter()

def summarize_pdf(file):
    loader = PyPDFLoader(file.name)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary

# create a function called generate_anki_cards that takes as input a pdf file and returns a list of anki cards using langchain LLMChain
def generate_anki_cards(file):
    ff=file.name
    # loader = PyPDFLoader(file.name)
    # docs = loader.load_and_split()

    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
    
    res = qa_document_chain.run(input_document=ff, question="Please generate Anki flashcards from the provided text. The format of each card should be a question;answer next line question;answer etc. Please generate 10 cards in total. The questions should cover important concepts and details from the document, and the answers should be concise and accurate. Thank you!")
    # print("\n\n------------------------ filename ------------------------\n\n")
    # print(file.name)
    # print("\n\n------------------------ qa_document_chain ------------------------\n\n")
    # print(qa_document_chain)
    # print("\n\n------------------------ Res ------------------------\n\n")
    print(res)
    return res

inputs = gr.File(label="Upload PDF")
outputs = gr.Textbox()

gr.Interface(fn=generate_anki_cards, inputs=inputs, outputs=outputs, title="PDF to Text Converter").launch()

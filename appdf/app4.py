from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
#import OPENAI_API_KEY from .env.local
from dotenv import load_dotenv
load_dotenv('.env.local')

reader = PdfReader('data/Lecture_4_old.pdf')
# print(reader)

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
# print(raw_text[:100])

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# print(len(texts))
# print(texts[0])
# print(texts[1])

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

# print(docsearch)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "Please generate Anki flashcards from the provided document. The format of each card should be: question;answer next line question;answer next line etc. Please generate 5 cards in total. The questions should cover important concepts and details from the document, and the answers should be concise and accurate. Thank you!"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query))
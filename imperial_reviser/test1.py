from langchain.llms import OpenAI  # import the OpenAI LLM

llm = OpenAI(temperature=0.9)  # high temperature means more randomness

# some input
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))

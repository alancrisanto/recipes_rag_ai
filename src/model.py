import os
from pinecone import Pinecone as pcn
from langchain_openai import ChatOpenAI 
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


api_key = os.environ.get("PINECONE_API_KEY")
openai_api_key = os.getenv('OPENAI_API_KEY')

# configure client
pc = pcn(api_key=api_key)
index_name = 'recipes-index'
index = pc.Index(index_name)
index.describe_index_stats()

# configure embeddings
embed_model = "text-embedding-3-small"
embed = OpenAIEmbeddings(model=embed_model)


# configure vector store
text_field = "description"
vectorStore = PineconeVectorStore(
  index, embed, text_field
)

# Create a retriever
retriever = vectorStore.as_retriever(search_type="similarity", k=3)

# Create an LLM
llm = ChatOpenAI(  
    model_name='gpt-4o-mini',  
    temperature=0.0  
) 


def get_recipe(query):
  """
  The query can be a string like "chicken recipe". This function will
  return the recipe that matches the query the most.

  Args:
    query (str): The query to search for.

  Returns:
    str: The recipe that matches the query the most.
  """
  # 1. Create a prompt
  system_prompt = (
    """
      Always add a kind and friendly message according to the context at the beginning of the response.
      Take the information from context[0]["metadata"]
      Then follow this format
      Title: [Title of the Recipe]
      Description: [Short Description of the Recipe]
      Preparation Time: [Preparation Time]
      Cooking Time: [Cooking Time]
      Difficulty: [Difficulty Level]
      Serves: [Number of Servings]
      Diet Type: [Diet Type]
      Nutrition: calories context[0]["metadata"]["calories"] | fat context[0]["metadata"]["fat"] | protein context[0]["metadata"]["protein"] | fiber context[0]["metadata"]["fibre"]
      Ingredients: [List of Ingredients]
      Instructions: [Step-by-step Cooking Instructions]

      If you don't have any general information, just respond with "I don't know!"

      {context}
      {{metadata}}
      """
  )

  # 2. Create a chain
  prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system_prompt),
          ("human", "{input}"),
      ]
  )

  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)


  # 3. Run the chain
  response = rag_chain.invoke({"input": query})
  return response["answer"]


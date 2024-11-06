import os
import json
import functions_framework
import google.cloud.logging
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.chains import LLMSummarizationCheckerChain
import vertexai
from vertexai.language_models import TextGenerationModel
from langchain.document_loaders import WebBaseLoader, DataFrameLoader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain



from functools import partial
from operator import itemgetter
from langchain.callbacks.manager import trace_as_chain_group
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document

PROJECT_ID  = os.environ.get('GCP_PROJECT','njlm-demo-2023')
LOCATION = os.environ.get('GCP_REGION','us-central1')

client = google.cloud.logging.Client(project=PROJECT_ID)
client.setup_logging()

log_name = "predictText-cloudfunction-log"
logger = client.logger(log_name)

def stuff_summary(docs):
    llm = ChatVertexAI(max_output_tokens=1024, temperature=0)

    prompt_template = """Summarize the context ensuring you generate an accurate, cohesive, and easy to read summary. Include all essential details while maintaining the truthfulness of the content, and rely solely on the information provided the text below:
    {text}
    Factual Summary:"""

    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text",output_key="output_text")

    result = stuff_chain.run(docs)

    return result

def refine_summary(docs):

    llm = ChatVertexAI(max_output_tokens=1024, temperature=0)

    prompt_template = """Summarize the context ensuring you generate an accurate, cohesive, and easy to read summary. Include all essential details while maintaining the truthfulness of the content, and rely solely on the information provided the text below:
     {text}
     Factual Summary:"""
    
    prompt = PromptTemplate.from_template(prompt_template)    
 
    refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary be factual and only use the new context and existing summary."
    "If the context isn't useful, return the original summary."
     )
    
    refine_prompt = PromptTemplate.from_template(refine_template)

    chain = load_summarize_chain(
     llm=llm,
     chain_type="refine",
     question_prompt=prompt,
     refine_prompt=refine_prompt,
     return_intermediate_steps=False,
     input_key="input_documents",
     output_key="output_text",
     )
    
    result = chain({"input_documents": docs}, return_only_outputs=True)

    return result


@functions_framework.http
def predictText(request):

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    request_json = request.get_json(silent=True)

    if request_json:
        command = request_json.get("command", "") 
        prompt_template = request_json.get("prompt", "") 
        highlightedTexts = request_json.get("highlightedTexts", "") 

        df = pd.DataFrame(highlightedTexts, columns=['highlightedTexts'])     
        loader = DataFrameLoader(df, page_content_column="highlightedTexts") 
        
        documents = loader.load() 
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5) 
        docs = text_splitter.split_documents(documents)   

        num_characters = len(highlightedTexts)
        result = ""

        if(num_characters < 500):
            result = stuff_summary(docs)   
        else:
            result = refine_summary(docs)      

    return json.dumps({"result": result })



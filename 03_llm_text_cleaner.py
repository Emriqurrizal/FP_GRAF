import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from config import OPENROUTER_API_KEY

# Ensure API key is available
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment or .env file.")

# Configure the LLM using ChatOpenAI but pointing to OpenRouter's API
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="moonshotai/kimi-k2", # Kimi by Moonshot
    max_tokens=1000,
    temperature=0.0
)

# Define the desired JSON structure for the LLM output
class DiseaseInfo(BaseModel):
    disease_name: str = Field(description="The normalized name of the disease")
    cleaned_description: str = Field(description="A concise, medical summary of the disease description")
    extracted_symptoms: list[str] = Field(description="A list of symptoms explicitly mentioned in the text")

# Setup the JSON output parser
parser = JsonOutputParser(pydantic_object=DiseaseInfo)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert medical data extraction assistant. Your job is to extract and clean unstructured medical text. "
               "Output MUST be in valid JSON format corresponding to the instructions.\n{format_instructions}"),
    ("human", "Extract information from the following raw medical text:\n\n{raw_text}")
])

# Create the chain
cleaner_chain = prompt | llm | parser

def clean_medical_text(raw_text: str) -> dict:
    """
    Passes raw medical text through the LLM to extract structured DiseaseInfo.
    """
    print(f"Parsing text using Kimi LLM: {raw_text[:50]}...")
    try:
        result = cleaner_chain.invoke({
            "raw_text": raw_text,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        print(f"Error parsing text: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    sample_text = (
        "Patient presents with severe Headache and a mild fever. The condition is known as "
        "Common Cold, which is a viral infection of the upper respiratory tract."
    )
    
    cleaned_data = clean_medical_text(sample_text)
    print("\nExtraction Result:")
    print(json.dumps(cleaned_data, indent=2))

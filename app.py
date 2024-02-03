from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain_community.chat_models import ChatOpenAI
# import time
import os
import requests


load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#img to text
def img2text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")
    
    text = image_to_text(url)[0]['generated_text']
    
    print(text)
    
    return text
img2text("photo.webp")

#llm
def generate_story(scenario):
    template="""
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words
    
    CONTEXT :{scenario}
    STORY:
    """
    # time.sleep(2) 
    prompt = PromptTemplate(template=template, input_variables = ["scenario"])
    
    story_llm = LLMChain(llm=OpenAI(
        model_name = "gpt-3.5-turbo", temperature = 1
    ), prompt=prompt, verbose = True)


    story = story_llm.predict(scenario=scenario)
    
    print(story)
    return story
#text to speech

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}
    payloads = {
        "inputs" : message
        
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac','wb') as file:
        file.write(response.content)



scenario = img2text("photo.webp")
story = generate_story(scenario)
text2speech(story)

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# audio, sampling_rate = query({
# 	"inputs": "The answer to the universe is 42",
# })
# # You can access the audio with IPython.display for example
# from IPython.display import Audio
# Audio(audio, rate=sampling_rate)
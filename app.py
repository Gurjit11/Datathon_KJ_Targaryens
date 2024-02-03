import os
from flask import Flask, request, jsonify, json, send_file
from waitress import serve
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
from dotenv import find_dotenv, load_dotenv
from duckduckgo_search import DDGS

# from langchain_community.chat_models import ChatOpenAI
# import time
import os
import requests
from pytube import YouTube
import tempfile
import openai
from flask_cors import CORS, cross_origin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import numpy as np


# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Research Agents"
os.environ["LANGCHAIN_API_KEY"] = "ls__852bb8fd93c4438ea4a09012c9410088"
load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Set the OpenAI API key
openai.api_key = os.environ.get(
    "API_KEY", "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"
)

app = Flask(__name__)
CORS(app)

data = "This video shows every YouTube play button ever handed out. Wow, okay, so apparently everyone on Earth has a silver play button. This looks like an army. 300,000, what? Gold play button. To me, if you have this, it basically means you make a comfortable living doing YouTube. Let's see how many people make a decent living as a YouTuber. 29,000, big drop-off from 300k. Okay, diamond play button. These ones look expensive. I imagine YouTube takes a hit to their bank account every time they send one of these out. Under a thousand. 50 mil, these are the A-list YouTubers, I guess you could say. 40 channels, that's not a lot. Red diamond award. I remember when PewDiePie first hit 100 mil, he basically made everyone realize it was even possible to do that. T-Series is an unstoppable force, they're probably gonna hit a billion subs in the future."


# pinecone
# Check to see if there is an environment variable with you API keys, if not, use what you put below
OPENAI_API_KEY = os.environ.get(
    "API_KEY", "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"
)

PINECONE_API_KEY = os.environ.get(
    "PINECONE_API_KEY", "d7868a9a-40af-426a-940a-84a4b01d6960"
)
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV", "gcp-starter")
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV,  # next to api key in console
)
index_name = "transcribe"  # put in the name of your pinecone index here
index = pinecone.Index(index_name)


docsearch = Pinecone

print("http://localhost:5000/")


@app.route("/", methods=["GET"])
@cross_origin()
def indexs():
    return f"hello {os.environ.get('SECRET_KEY')}"


# @app.route("/textaudio", methods=["POST"])
# @cross_origin()
# def textaudio():
#     request_data = json.loads(request.data)
#     message = request_data["text"]
#     API_URL = (
#         "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
#     )
#     headers = {"Authorization": "Bearer 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"}
#     payloads = {"inputs": message}

#     response = requests.post(API_URL, headers=headers, json=payloads)
#     with open("audio.flac", "wb") as file:
#         file.write(response.content)

#     return send_file("audio.flac", mimetype="audio/flac", as_attachment=True)


@app.route("/textprompt", methods=["POST"])
@cross_origin()
def textprompt():
    request_data = json.loads(request.data)
    text = request_data["text"]
    print(text)
    if not text:
        return jsonify({"error": "Video URL is required."}), 400

    try:

        def internet_search(query: str) -> str:
            """Searches the internet using DuckDuckGo."""
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                return results if results else "No results found."

        result = internet_search(text)
        combined_body = " ".join([item["body"] for item in result])
        print(combined_body)

        def generate_story(scenario):
            template = """
            You are a report maker;
            You can generate a short report based on a simple narrative, the final news like report should be no more than 150 words
            
            CONTEXT :{scenario}
            REPORT:
            """
            # time.sleep(2)
            prompt = PromptTemplate(template=template, input_variables=["scenario"])

            story_llm = LLMChain(
                llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1),
                prompt=prompt,
                verbose=True,
            )

            story = story_llm.predict(scenario=scenario)

            print(story)
            return story

        story = generate_story(combined_body)

        return jsonify({"script": story}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/imageprompt", methods=["POST"])
@cross_origin()
def imageprompt():
    request_data = json.loads(request.data)
    url = request_data["url"]
    print(url)

    def img2text(url):
        image_to_text = pipeline(
            "image-to-text", model="Salesforce/blip-image-captioning-base"
        )
        text = image_to_text(
            "https://media.istockphoto.com/id/1493866295/photo/support-the-wall-with-the-palm.jpg?s=2048x2048&w=is&k=20&c=_AqF3cnsA1bcnE91VkOYueaa8JV5XugbWILuWv5arCI="
        )[0]["generated_text"]
        print(text)
        return text

    if not url:
        return jsonify({"error": "Video URL is required."}), 400
    try:
        text = img2text(url)
        return jsonify({"script": text + "hii"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route("/transcribe", methods=["POST"])
# @cross_origin()
# def transcribe():
#     request_data = json.loads(request.data)
#     video_url = request_data["video_url"]

#     if not video_url:
#         return jsonify({"error": "Video URL is required."}), 400

#     try:
#         # Download YouTube video
#         yt = YouTube(video_url)
#         stream = yt.streams.filter(only_audio=True).first()

#         # Save the video in a temporary file
#         with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
#             temp_file_name = temp_file.name

#         stream.download(
#             output_path=os.path.dirname(temp_file_name),
#             filename=os.path.basename(temp_file_name),
#         )

#         # Transcribe the video using OpenAI's Whisper API
#         with open(temp_file_name, "rb") as audio_file:
#             transcript = openai.Audio.transcribe("whisper-1", audio_file)

#         # Remove the temporary file after transcription
#         os.remove(temp_file_name)

#         # splitting
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
#         texts = text_splitter.create_documents([transcript["text"]])
#         print(len(texts))

#         index_stats_response = index.describe_index_stats()
#         print(index_stats_response)

#         embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#         global docsearch
#         docsearch = Pinecone.from_texts(
#             [t.page_content for t in texts], embeddings, index_name=index_name
#         )

#         return jsonify({"script": transcript["text"]}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/query", methods=["POST"])
# @cross_origin()
# def query():
#     request_data = json.loads(request.data)
#     query = request_data["query"]

#     if not query:
#         return jsonify({"error": "Question is required."}), 400

#     try:
#         # openai
#         llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
#         chain = load_qa_chain(llm, chain_type="stuff")

#         docs = docsearch.similarity_search(query)
#         print(docs)

#         output = chain.run(input_documents=docs, question=query)
#         print(output)

#         return jsonify({"ans": str(output)}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/clear", methods=["POST"])
# @cross_origin()
# def clear():
#     try:

#         def get_ids_from_query(index, input_vector):
#             print("searching pinecone...")
#             results = index.query(
#                 vector=input_vector, top_k=10000, include_values=False
#             )
#             ids = set()
#             print(type(results))
#             for result in results["matches"]:
#                 ids.add(result["id"])
#             return ids

#         def get_all_ids_from_index(index, num_dimensions, namespace=""):
#             num_vectors = index.describe_index_stats()["namespaces"][namespace][
#                 "vector_count"
#             ]
#             all_ids = set()
#             while len(all_ids) < num_vectors:
#                 print(
#                     "Length of ids list is shorter than the number of total vectors..."
#                 )
#                 input_vector = np.random.rand(num_dimensions).tolist()
#                 print("creating random vector...")
#                 ids = get_ids_from_query(index, input_vector)
#                 print("getting ids from a vector query...")
#                 all_ids.update(ids)
#                 print("updating ids set...")
#                 print(f"Collected {len(all_ids)} ids out of {num_vectors}.")

#             return all_ids

#         all_ids = get_all_ids_from_index(index, num_dimensions=1536, namespace="")
#         print(list(all_ids))

#         delete_response = index.delete(ids=list(all_ids), namespace="")
#         print(delete_response)
#         return jsonify({"ans": str("Cleared DB successfully!")}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000, threads=2)

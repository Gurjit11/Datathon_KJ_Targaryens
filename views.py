import os
from flask import request, jsonify, json
from pytube import YouTube
import tempfile
import openai
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import numpy as np

# from app import app
from flask import Flask

# Inject Flask magic
app = Flask(__name__)

# Set the OpenAI API key
openai.api_key = os.environ.get(
    "API_KEY", "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"
)

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


@app.route("/", methods=["GET"])
def indexs():
    return f"hello {os.environ.get('SECRET_KEY')}"


@app.route("/transcribe", methods=["POST"])
def transcribe():
    request_data = json.loads(request.data)
    video_url = request_data["video_url"]

    if not video_url:
        return jsonify({"error": "Video URL is required."}), 400

    try:
        # Download YouTube video
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()

        # Save the video in a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file_name = temp_file.name

        stream.download(
            output_path=os.path.dirname(temp_file_name),
            filename=os.path.basename(temp_file_name),
        )

        # Transcribe the video using OpenAI's Whisper API
        with open(temp_file_name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

        # Remove the temporary file after transcription
        os.remove(temp_file_name)

        # splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        texts = text_splitter.create_documents([transcript["text"]])
        print(len(texts))

        index_stats_response = index.describe_index_stats()
        print(index_stats_response)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        global docsearch
        docsearch = Pinecone.from_texts(
            [t.page_content for t in texts], embeddings, index_name=index_name
        )

        return jsonify({"script": transcript["text"]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    request_data = json.loads(request.data)
    query = request_data["query"]

    if not query:
        return jsonify({"error": "Question is required."}), 400

    try:
        # openai
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")

        docs = docsearch.similarity_search(query)
        print(docs)

        output = chain.run(input_documents=docs, question=query)
        print(output)

        return jsonify({"ans": str(output)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    try:

        def get_ids_from_query(index, input_vector):
            print("searching pinecone...")
            results = index.query(
                vector=input_vector, top_k=10000, include_values=False
            )
            ids = set()
            print(type(results))
            for result in results["matches"]:
                ids.add(result["id"])
            return ids

        def get_all_ids_from_index(index, num_dimensions, namespace=""):
            num_vectors = index.describe_index_stats()["namespaces"][namespace][
                "vector_count"
            ]
            all_ids = set()
            while len(all_ids) < num_vectors:
                print(
                    "Length of ids list is shorter than the number of total vectors..."
                )
                input_vector = np.random.rand(num_dimensions).tolist()
                print("creating random vector...")
                ids = get_ids_from_query(index, input_vector)
                print("getting ids from a vector query...")
                all_ids.update(ids)
                print("updating ids set...")
                print(f"Collected {len(all_ids)} ids out of {num_vectors}.")

            return all_ids

        all_ids = get_all_ids_from_index(index, num_dimensions=1536, namespace="")
        print(list(all_ids))

        delete_response = index.delete(ids=list(all_ids), namespace="")
        print(delete_response)
        return jsonify({"ans": str("Cleared DB successfully!")}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

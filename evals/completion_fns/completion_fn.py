from ctypes import util
from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import ChatCompletionPrompt, CompletionPrompt
from evals.record import record_sampling
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import openai

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = "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"
# Configuration de l'API OpenAI
openai.api_key = "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"
# Configuration de sentence-transformers
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

mongo_client = MongoClient("mongodb://testaigpt:vNzNb9VasogHq3vKMDH3y6Gwcv7ghQmqcJb6VFUgdkGk0IG20d9Gm3lcfmC1PizKl9oNgRnQR1i2ACDbukkr8Q%3D%3D@testaigpt.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@testaigpt@", retryWrites=False)

class MyRetrievalCompletionResult(CompletionResult):
    def __init__(self, response: str) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]

class MyRetrievalCompletionFn(CompletionFn):
    def __init__(
        self,
        model,
        mongo_client,
        retrieval_template: str = "Use the provided context to answer the question. ",
        k: int = 5,
        **kwargs
    ) -> None:
        self.model = model
        self.mongo_client = mongo_client
        self.retrieval_template = retrieval_template
        self.k = k

    def __call__(self, prompt, **kwargs) -> MyRetrievalCompletionResult:
        
        question = CompletionPrompt(prompt).to_formatted_prompt()
        db = mongo_client['knowledge']
        collection = db['zhc_roi']

        question_embedding = model.encode(question, convert_to_tensor=True)
        cursor = collection.find({})
        similarities = []

        for doc in cursor:
            fragment_embedding = model.encode(doc['paragraph'], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(question_embedding, fragment_embedding)
            similarities.append((similarity, doc['paragraph'], doc['page']))

        sorted_fragments = sorted(similarities, key=lambda x: x[0], reverse=True)
        top_fragments = [frag[1] for frag in sorted_fragments[:self.k]]

        # Build prompt manually without Flask app dependencies
        system_content = self.retrieval_template + '\n'.join(top_fragments)
        retrieval_prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=retrieval_prompt,
            api_key=OPENAI_API_KEY
        )

        answer = response['choices'][0]['message']['content']
        record_sampling(prompt=retrieval_prompt, sampled=answer)
        return MyRetrievalCompletionResult(answer)

# Assume you have initialized your model and MongoDB client


# Initialize the completion function with your model and MongoDB client
retrieval_completer = MyRetrievalCompletionFn(model, mongo_client)

from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.config_loader import settings


class ChatModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name == "gpt-3.5-turbo":
            self.model = "put_your_gpt35_here"
        elif model_name == "gpt-4":
            self.model = "put_your_gpt4_here"
        elif model_name == "gemini-2.5-pro":
            self.model = ChatGoogleGenerativeAI(model="gemini-2.5-pro",api_key=settings.GOOGLE_API_KEY)

    def format_prompt(self, query: str, context: str) -> str:
        prompt=f""""i have provided you the context and the question. look the contex carefully and give the correct answer 
        for the given question. dont provide any extra information other than the answer.
        if you dont find the answer in the context, simply reply with 'i dont know the answer'. and please add the preamble according to user question
        context:{context} \n
        question:{query}"""
        return prompt
    
    def generate_response(self, prompt: str) -> str:
       final_response = self.model.invoke(prompt)
       return final_response
    
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

load_dotenv()

class LLM:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1, max_tokens: int = 200):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not self.api_key:
            raise ValueError("API Key não foi fornecida nem encontrada em OPENAI_API_KEY.")
        self.client = ChatOpenAI(model=self.model, api_key=self.api_key, temperature=self.temperature, max_tokens=self.max_tokens)

    def ask(self, text: str):
        try:
            prompt = PromptTemplate.from_template("Responda a pegunta a seguir da forma "
                                                  "mais direta possivel sem passar informações "
                                                  "desnecessárias para o usuário. Pergunta: {text}")
            prompt = prompt.format(text=text)
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Erro ao interagir com o modelo: {e}"

    def summarize(self, text: str):
        prompt = PromptTemplate.from_template("Resuma o seguinte texto:\n\n{text}")
        prompt = prompt.format(text=text)
        return self.ask(prompt)

    def translate(self, text: str, target_language: str = "English"):
        prompt = PromptTemplate.from_template("Traduza o seguinte texto para {target_language}:\n\n{text}")
        prompt = prompt.format(target_language=target_language, text=text)
        return self.ask(prompt)
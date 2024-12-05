from flask import Flask, request, jsonify
from llm_openai import LLM

app = Flask(__name__)
llm = LLM()

@app.route('/chat', methods=['POST'])
def chat():
    data    = request.get_json()
    message = data.get('message')
    response = llm.ask(message)

    return jsonify({ "message": response })
    
    
if __name__ == '__main__':
    app.run(port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from code_assistant import CodeAssistant

app = Flask(__name__)
CORS(app)  # This will enable CORS for all domains

assistant = CodeAssistant()

@app.route('/generate', methods=['POST'])
def generate():
    user_content = request.json.get('content')
    if not user_content:
        return jsonify({"error": "No content provided"}), 400
    
    response = assistant.generate_response(user_content)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify, render_template
from chatbot.qwen_model import get_chatbot_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print("[INFO] Data received from frontend:", data)

        user_message = data.get("message", "").strip()
        location = data.get("location", "Unknown location").strip()

        if not user_message:
            return jsonify({"response": "⚠️ Please enter a message."}), 400

        response = get_chatbot_response(user_message, location)
        print("[INFO] Response from model:", response)

        # Ensure HTML-safe and formatted response for frontend
        formatted_response = response.replace("\n", "<br>").strip()

        return jsonify({"response": formatted_response})

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"response": f"⚠️ Internal error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

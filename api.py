import sys, os
from unittest import result
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Core'))

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from nlp_evaluator import AIEvaluator

app = Flask(__name__)
CORS(app)

evaluator = AIEvaluator()

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.route('/evaluate', methods=['POST', 'OPTIONS'])
def evaluate():
    if request.method == 'OPTIONS':
        return make_response('', 200)
    try:
        data = request.json
        print("Received:", data)
        result = evaluator.evaluate_answer(
            data.get('ideal_answer', ''),
            data.get('candidate_answer', ''),
            data.get('is_behavioral', False)
        )
        result['score_out_of_10'] = float(result['score_out_of_10'])
        return jsonify(result)
    
    except Exception as e:
        print("ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
import joblib
from konlpy.tag import Okt
from flask import Flask, request, jsonify, render_template

app = Flask(__name__) 

    # 터미널 표시
#  * Running on all addresses (0.0.0.0) / 이거 안하면, 
#  * Running on http://127.0.0.1:5001
#  * Running on http://10.125.121.172:5001

# 반드시 컴퓨터에 자바가 깔려 있어야 함.
okt = Okt()

def tw_tokenzier(text):
    tokenzier_ko = okt.morphs(text)
    return tokenzier_ko

try:
    model = joblib.load("lr_v1.pkl")
    vec = joblib.load("tfidf_vect.pkl")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {str(e)}")
    raise


@app.route("/")
def home(): # 1. 내가 설치한 플라스크 서버가 잘 돌아가는지 확인 하려고
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # < 웹개발 부분>
    data = request.get_json() # 요청하는 정보를 바꿔줌
    if not data or "text" not in data:
        return jsonify({"error": "텍스트가 올바르지 않거나 제공되지 않습니다."}), 400
    
    text = data["text"]
    if not text.strip():
        return jsonify({"error": "텍스트가 올바르지 않거나 제공되지 않습니다."}), 400
    
    # <데이터 분석 부분>
    #vec를 transform하고
    #lr을 사용해서 예측하고
    #긍정/부정으로 변경해서
    text_tfidf = vec.transform([text])
    predict = model.predict(text_tfidf)[0]
    
    return jsonify({"emotion" : str(predict)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
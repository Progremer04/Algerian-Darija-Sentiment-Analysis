# app.py
from flask import Flask, render_template_string, request, jsonify
import torch
import torch.nn as nn
import json
import re
import os

app = Flask(__name__)

# ============================================================
# 🧠 MODEL
# ============================================================
class LSTMCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, lstm_hidden=128, conv_filters=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(2*lstm_hidden, conv_filters, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(conv_filters, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.conv(x.transpose(1,2))
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x).squeeze(-1)

# ============================================================
# 🔧 TEXT UTILS
# ============================================================
def clean(text):
    text = text.replace("<br/>", " ")
    strip_special_chars = re.compile(u'[^\u0621-\u064a ]')
    return re.sub(strip_special_chars, " ", text)

def process(text):
    text = re.sub('\ـ+', ' ', text)
    text = re.sub('\ر+', 'ر', text)
    text = re.sub('\اا+','ا',text)
    text = re.sub('\ووو+','و',text)
    text = re.sub('\ههه+','ههه',text)
    text = re.sub('\ةة+','ة',text)
    text = re.sub('\ييي+','ي',text)
    text = re.sub('أ','ا',text)
    text = re.sub('آ','ا',text)
    text = re.sub('إ','ا',text)
    text = re.sub('ة','ه',text)
    text = re.sub('ى','ي',text)
    text = " ".join(text.split())
    return text

def enhanced_clean_text(text):
    if not isinstance(text, str): return ""
    text = clean(text)
    text = process(text)
    return text

def encode_text(text, vocab, max_len):
    tokens = text.split()
    encoded = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded

# ============================================================
# 📥 LOAD MODEL + VOCAB
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
max_len = metadata.get("max_len", 100)

with open('vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

model = LSTMCNN(len(vocab))
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

# ============================================================
# 🌐 FLASK APP
# ============================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SUPER Arabic Sentiment</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gradient-to-br from-purple-50 to-blue-100 min-h-screen">
<div class="container mx-auto px-4 py-8">
    <!-- Header -->
    <div class="text-center mb-12">
        <div class="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-r from-purple-500 to-blue-600 rounded-full shadow-xl mb-6">
            <i class="fas fa-rocket text-white text-4xl"></i>
        </div>
        <h1 class="text-5xl font-bold text-gray-800 mb-3">SUPER Arabic Sentiment</h1>
        <p class="text-xl text-gray-600 mb-2">Enhanced AI-powered sentiment detection</p>
        <div class="inline-flex items-center bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm font-semibold">
            <i class="fas fa-check-circle mr-2"></i> LSTMCNN Model Loaded
        </div>
    </div>

    <div class="max-w-4xl mx-auto">
        <!-- Input Card -->
        <div class="bg-white rounded-2xl shadow-xl p-8 mb-8 transition-all duration-300 hover:shadow-2xl">
            <div class="mb-6">
                <label class="block text-lg font-semibold text-gray-700 mb-3">
                    <i class="fas fa-keyboard mr-2 text-purple-500"></i>Enter Arabic Text
                </label>
                <textarea id="textInput" rows="4" class="w-full px-4 py-4 border-2 border-gray-200 rounded-xl focus:border-purple-500 focus:ring-2 focus:ring-purple-200 resize-none text-lg transition-all duration-300" placeholder="اكتب النص العربي هنا..."></textarea>
                <div class="flex justify-between items-center mt-2">
                    <span class="text-sm text-gray-500">Advanced Arabic text processing</span>
                    <span id="charCount" class="text-sm text-gray-500">0 characters</span>
                </div>
            </div>
            <button id="analyzeBtn" class="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-bold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 flex items-center justify-center text-lg shadow-lg">
                <i class="fas fa-brain mr-3"></i> Analyze with SUPER AI
            </button>
        </div>

        <!-- Result Card -->
        <div id="resultSection" class="hidden bg-white rounded-2xl shadow-xl p-8 mb-8 animate__animated">
            <h3 class="text-2xl font-bold text-gray-800 mb-6 flex items-center"><i class="fas fa-chart-pie mr-3 text-purple-500"></i> SUPER Analysis Results</h3>
            <div class="grid md:grid-cols-2 gap-6 mb-6">
                <div class="p-4 bg-gray-50 rounded-xl border-2 border-gray-100"><p class="text-gray-600 mb-2 font-semibold"><i class="fas fa-language mr-2"></i>Original Text</p><p id="originalText" class="text-gray-800 text-lg min-h-[60px]"></p></div>
                <div class="p-4 bg-gray-50 rounded-xl border-2 border-gray-100"><p class="text-gray-600 mb-2 font-semibold"><i class="fas fa-broom mr-2"></i>Processed Text</p><p id="cleanedText" class="text-gray-800 text-lg text-right min-h-[60px]" dir="rtl"></p></div>
            </div>
            <div id="result" class="text-center"><!-- Result --></div>

            <!-- Confidence Meter -->
            <div class="mt-8">
                <p class="text-gray-700 font-semibold mb-3 flex items-center justify-center"><i class="fas fa-bullseye mr-2"></i> AI Confidence Level</p>
                <div class="w-full bg-gray-200 rounded-full h-6 mb-2 relative">
                    <div id="confidenceBar" class="h-6 rounded-full transition-all duration-2000 ease-out relative">
                        <div class="absolute right-0 -top-8 bg-gray-800 text-white px-2 py-1 rounded text-sm font-bold" id="confidenceText">0%</div>
                    </div>
                </div>
                <div class="flex justify-between text-sm text-gray-600"><span>Low Confidence</span><span>High Confidence</span></div>
            </div>
        </div>

        <!-- Demo Section -->
        <div class="bg-white rounded-2xl shadow-xl p-8">
            <h3 class="text-xl font-bold text-gray-800 mb-6 flex items-center"><i class="fas fa-flask mr-3 text-purple-500"></i> SUPER Test Examples</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button class="demo-btn p-4 bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 rounded-xl hover:from-purple-100 hover:to-blue-100 border-2 border-purple-100 transition-all duration-300 text-right transform hover:scale-105" dir="rtl"><i class="fas fa-play-circle ml-2 text-purple-500"></i> واش راك اليوم؟ الخدمة مليحة والحمد لله</button>
                <button class="demo-btn p-4 bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 rounded-xl hover:from-purple-100 hover:to-blue-100 border-2 border-purple-100 transition-all duration-300 text-right transform hover:scale-105" dir="rtl"><i class="fas fa-play-circle ml-2 text-purple-500"></i> ياكلو في حق شعي</button>
                <button class="demo-btn p-4 bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 rounded-xl hover:from-purple-100 hover:to-blue-100 border-2 border-purple-100 transition-all duration-300 text-right transform hover:scale-105" dir="rtl"><i class="fas fa-play-circle ml-2 text-purple-500"></i> هذا أفضل يوم في حياتي</button>
                <button class="demo-btn p-4 bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 rounded-xl hover:from-purple-100 hover:to-blue-100 border-2 border-purple-100 transition-all duration-300 text-right transform hover:scale-105" dir="rtl"><i class="fas fa-play-circle ml-2 text-purple-500"></i> ماعنديش فلوس والوضعية صعيبة</button>
            </div>
        </div>
    </div>

    <footer class="text-center mt-12 text-gray-500 text-sm">
        <p>Powered by SUPER LSTMCNN • Enhanced Arabic Sentiment Analysis</p>
    </footer>
</div>

<script>
document.addEventListener('DOMContentLoaded', function(){
    const analyzeBtn=document.getElementById('analyzeBtn');
    const textInput=document.getElementById('textInput');
    const resultSection=document.getElementById('resultSection');
    const originalText=document.getElementById('originalText');
    const cleanedText=document.getElementById('cleanedText');
    const resultDiv=document.getElementById('result');
    const confidenceBar=document.getElementById('confidenceBar');
    const confidenceText=document.getElementById('confidenceText');
    const demoButtons=document.querySelectorAll('.demo-btn');
    const charCount=document.getElementById('charCount');

    textInput.addEventListener('input',()=>charCount.textContent=textInput.value.length+' characters');

    analyzeBtn.addEventListener('click',()=>{
        const text=textInput.value.trim();
        if(!text){showNotification('Please enter some text','error');return;}
        analyzeBtn.disabled=true;
        analyzeBtn.innerHTML='<i class="fas fa-spinner fa-spin mr-2"></i> SUPER AI Analyzing...';
        fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
        .then(r=>r.json()).then(data=>{
            analyzeBtn.disabled=false;
            analyzeBtn.innerHTML='<i class="fas fa-brain mr-3"></i> Analyze with SUPER AI';
            if(data.error){showNotification('Error: '+data.error,'error');return;}
            displayResults(data);
        }).catch(err=>{
            analyzeBtn.disabled=false;
            analyzeBtn.innerHTML='<i class="fas fa-brain mr-3"></i> Analyze with SUPER AI';
            showNotification('Analysis failed','error');
        });
    });

    demoButtons.forEach(btn=>btn.addEventListener('click',()=>{
        textInput.value=btn.textContent.replace('▶','').trim();
        charCount.textContent=textInput.value.length+' characters';
        analyzeBtn.click();
    }));

    function displayResults(data){
        resultSection.classList.remove('hidden');
        resultSection.classList.add('animate__fadeIn');
        originalText.textContent=data.text;
        cleanedText.textContent=data.cleaned;
        const prob=data.probability;
        const isPositive=data.prediction==='Positive';
        const confidencePercent=isPositive?Math.round(prob*100):Math.round((1-prob)*100);
        confidenceBar.style.width=`${confidencePercent}%`;
        confidenceBar.style.background=isPositive?'linear-gradient(90deg, #10B981, #34D399)':'linear-gradient(90deg, #EF4444, #F87171)';
        confidenceText.textContent=confidencePercent+'%';
        confidenceText.style.color=isPositive?'#10B981':'#EF4444';
        const color=isPositive?'green':'red';
        const icon=isPositive?'fa-smile-beam':'fa-frown';
        const bgColor=isPositive?'bg-green-50 border-green-200':'bg-red-50 border-red-200';
        const textColor=isPositive?'text-green-800':'text-red-800';
        resultDiv.innerHTML=`
            <div class="${bgColor} border-2 rounded-2xl p-6 animate__animated animate__pulse">
                <div class="flex items-center justify-center space-x-6">
                    <i class="fas ${icon} text-5xl ${textColor}"></i>
                    <div class="text-center">
                        <h4 class="text-3xl font-bold ${textColor} mb-2">${data.prediction} SENTIMENT</h4>
                        <p class="text-gray-600 text-lg">Confidence: <span class="font-bold">${data.confidence}</span></p>
                        <p class="text-gray-500 mt-2">Probability: ${prob.toFixed(4)}</p>
                    </div>
                </div>
            </div>`;
        resultSection.scrollIntoView({behavior:'smooth',block:'start'});
    }
    function showNotification(msg,type){
        const existing=document.querySelector('.notification');
        if(existing)existing.remove();
        const note=document.createElement('div');
        note.className=`notification fixed top-4 right-4 p-4 rounded-xl shadow-2xl text-white font-bold z-50 animate__animated animate__fadeInRight ${type==='error'?'bg-red-500':'bg-green-500'}`;
        note.innerHTML=`<div class="flex items-center"><i class="fas ${type==='error'?'fa-exclamation-triangle':'fa-check-circle'} mr-3 text-xl"></i><span>${msg}</span></div>`;
        document.body.appendChild(note);
        setTimeout(()=>{note.classList.add('animate__fadeOutRight');setTimeout(()=>note.remove(),1000)},5000);
    }
});
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text.strip():
            return jsonify({'error': 'Please enter some text'}), 400
        cleaned = enhanced_clean_text(text)
        ids = torch.tensor([encode_text(cleaned, vocab, max_len)]).to(device)
        with torch.no_grad():
            output = model(ids)
            prob = torch.sigmoid(output).item()
        pred = "Positive" if prob >= 0.5 else "Negative"
        confidence = f"{prob:.2%}" if pred == "Positive" else f"{1-prob:.2%}"
        return jsonify({
            'text': text,
            'cleaned': cleaned,
            'prediction': pred,
            'confidence': confidence,
            'probability': prob
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 SUPER Arabic Sentiment Analysis Web App Started!")
    print("📍 Access at: http://localhost:5000")
    app.run(debug=True)

import os
import numpy as np
from flask import Flask, request, render_template
import pickle
import google.generativeai as genai
import PIL.Image
import json
from dotenv import load_dotenv  # <--- NEW IMPORT

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# 1. Setup the API Key from the .env file
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# 2. The Brain
model_ai = genai.GenerativeModel('gemini-flash-latest')
# ---------------------

# Create a dictionary with advice for each crop
crop_advice = {
    'rice': "Maximize Yield: Maintain 5-10cm water level. Quality Tip: Use organic manure for better grain filling.",
    'maize': "Maximize Yield: Ensure good drainage. Quality Tip: Harvest when kernels are hard for best storage.",
    'chickpea': "Maximize Yield: Avoid waterlogging. Quality Tip: Phosphate fertilizer improves grain size.",
    'kidneybeans': "Maximize Yield: Needs trellis support. Quality Tip: Harvest pods young for tenderness.",
    'pigeonpeas': "Maximize Yield: Intercrop with sorghum. Quality Tip: Dry thoroughly to prevent mold.",
    'mothbeans': "Maximize Yield: Highly drought resistant. Quality Tip: Avoid over-watering.",
    'mungbean': "Maximize Yield: Short duration crop. Quality Tip: Harvest before pods shatter.",
    'blackgram': "Maximize Yield: Use phosphorus rich fertilizer. Quality Tip: Store in dry conditions.",
    'lentil': "Maximize Yield: Grows well in residual moisture. Quality Tip: Weed control is crucial early on.",
    'pomegranate': "Maximize Yield: Regular pruning required. Quality Tip: Bag fruits to prevent pest attacks.",
    'banana': "Maximize Yield: Heavy feeder, needs potassium. Quality Tip: Use tissue-cultured plants.",
    'mango': "Maximize Yield: induce flowering with potassium nitrate. Quality Tip: Harvest at mature green stage.",
    'grapes': "Maximize Yield: Pruning determines yield. Quality Tip: Avoid moisture on berries to prevent rot.",
    'watermelon': "Maximize Yield: Mulching increases yield. Quality Tip: Stop watering 1 week before harvest for sweetness.",
    'muskmelon': "Maximize Yield: Warm climate crop. Quality Tip: Harvest at 'full slip' stage.",
    'apple': "Maximize Yield: Cross-pollination is key (plant different varieties). Quality Tip: Thin fruits for larger size.",
    'orange': "Maximize Yield: Micro-irrigation works best. Quality Tip: Potassium spray improves fruit finish.",
    'papaya': "Maximize Yield: Avoid water stagnation. Quality Tip: Harvest when skin turns 25% yellow.",
    'coconut': "Maximize Yield: Regular salt application helps. Quality Tip: Harvest every 45 days.",
    'cotton': "Maximize Yield: Pest management (Bollworm) is critical. Quality Tip: Pick dry cotton for best grade.",
    'jute': "Maximize Yield: Needs standing water for retting. Quality Tip: Harvest at small pod stage for fine fiber.",
    'coffee': "Maximize Yield: Shade regulation is key. Quality Tip: Selective picking of red berries only."
}

app = Flask(__name__, template_folder='website', static_folder='website/static')

# Load the ML Model
# Ensure this path is correct relative to where you run the script
model_path = os.path.join(os.path.dirname(__file__), 'ML_model', 'crop_prediction.pkl')

# Simple check to prevent crash if model is missing during testing
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None
    print(f"WARNING: Model file not found at {model_path}")

@app.route('/')
def home():
    return render_template('index.html')

# --- CROP PREDICTION ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: ML Model not found.")

    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['Rainfall'])
    except (KeyError, ValueError):
        return render_template('index.html', prediction_text='Invalid input: please enter numeric values.')

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    output = prediction[0]
    
    advice = crop_advice.get(output, "No specific advice available.")

    return render_template('index.html', prediction_text=f'Recommended Crop: {output}', advice_text=advice)

# --- DISEASE DETECTION ---
@app.route('/disease-predict', methods=['POST'])
def disease_predict():
    if 'file' not in request.files:
        return render_template('index.html', disease_result="No file uploaded.")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', disease_result="No file selected.")

    if file:
        try:
            img = PIL.Image.open(file)

            prompt = """
            Analyze this plant image. Identify the disease.
            Return a JSON object with:
            {
                "disease_name": "Name of the disease",
                "status": "Healthy or Infected",
                "remedy_organic": "One organic/home remedy description",
                "remedy_chemical": "One recommended chemical medicine/pesticide name",
                "prevention": "Short prevention tip"
            }
            """

            response = model_ai.generate_content([prompt, img])
            
            cleaned_text = response.text.replace("```json", "").replace("```", "")
            result_data = json.loads(cleaned_text)

            disease_msg = f"Detected: {result_data['disease_name']} ({result_data['status']})"
            remedy_msg = result_data['remedy_organic']
            chem_msg = result_data['remedy_chemical']
            
            return render_template('index.html', disease_result=disease_msg, organic_result=remedy_msg, chemical_result=chem_msg)
        
        except Exception as e:
            print(f"Error: {e}") 
            return render_template('index.html', disease_result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
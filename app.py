from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('/Users/safwan/car_price_prediction/car_price_predictor_linear.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract form data
        data = {
            'make': request.form.get('make'),
            'fuel': request.form.get('fuel'),
            'doors': request.form.get('doors'),
            'body': request.form.get('body'),
            'drive': request.form.get('drive'),
            'weight': float(request.form.get('weight')),
            'engine-size': float(request.form.get('engine_size')),
            'bhp': float(request.form.get('bhp')),
            'mpg': float(request.form.get('mpg'))
        }
        
        # Make a prediction
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        
        return render_template('index.html', prediction=f'${prediction:.2f}')
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

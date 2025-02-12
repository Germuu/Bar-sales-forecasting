from flask import Flask, render_template, request
import os
import pandas as pd
from src.services.file_handler import process_file
from src.models.predictor import predict_sales

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Home page - Upload CSV, Make predictions, and show results
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Handle file upload
        file = request.files.get('file')
        event_data = request.form.getlist('event_data')  # Get event data for the next 7 days
        
        if file and file.filename:
            # Save file and get filepath
            filepath = process_file(file, app.config['UPLOAD_FOLDER'])
            
            # Make predictions with event data
            predictions = predict_sales(filepath, event_data)
            
            # Restocking logic (mock current inventory and safety buffer for now)
            current_inventory = {'Beer': 200, 'Vodka': 100}  # Mock current stock for now
            safety_buffer = 0.10  # 10% safety buffer
            
            # Calculate order quantities
            restock_info = []
            for drink in predictions['Drink']:
                predicted_sales = predictions[predictions['Drink'] == drink]['Predicted Sales'].values[0]
                current_stock = current_inventory.get(drink, 0)
                order_quantity = max(predicted_sales - current_stock, 0) + (predicted_sales * safety_buffer)
                restock_info.append({
                    'Drink': drink,
                    'Predicted Sales': predicted_sales,
                    'Current Stock': current_stock,
                    'Order Quantity': round(order_quantity)
                })
            
            # Combine predictions and restocking info into a single table
            restock_df = pd.DataFrame(restock_info)
            
            # Render the result table with predictions and restocking info
            return render_template('index.html', tables=[predictions.to_html(classes='table table-bordered table-striped', index=False),
                                                        restock_df.to_html(classes='table table-bordered table-striped', index=False)],
                                   titles=predictions.columns.values)
    
    return render_template('index.html')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

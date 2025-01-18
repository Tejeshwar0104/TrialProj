from flask import Flask, request, render_template, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer
import fitz  # PyMuPDF
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('summarisation.html')  # Redirect to the summarisation page

# Initialize the model and tokenizer directly
model_name = "facebook/bart-large-cnn"  # You can replace this with your preferred BART model
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text

@app.route('/summarisation.html')
def summarisation_page():
    """Renders the summarisation HTML page."""
    return render_template('summarisation.html')

@app.route('/summarise', methods=['POST'])
def summarise_pdf():
    """Handles PDF summarisation requests."""
    if 'pdf' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    pdf_path = os.path.join("uploads", pdf_file.filename)
    try:
        pdf_file.save(pdf_path)
    except Exception as e:
        return jsonify({"error": f"File save error: {str(e)}"}), 500

    # Extract text from PDF
    document_text = extract_text_from_pdf(pdf_path)
    if not document_text.strip():
        return jsonify({"error": "Unable to extract text from the uploaded PDF."}), 400

    # Tokenize and summarize
    try:
        inputs = tokenizer.encode("summarize: " + document_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs, 
            max_length=150, 
            min_length=40, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return jsonify({"error": f"Summarisation error: {str(e)}"}), 500

    # Clean up uploaded file
    try:
        os.remove(pdf_path)
    except Exception as e:
        print(f"Error removing file: {e}")

    return jsonify({"summary": summary})

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)

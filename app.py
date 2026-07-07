import streamlit as st
import cv2
import numpy as np
import easyocr
import json
import re
import ollama
import fitz  # PyMuPDF: Handles PDFs with 0 external installations!
from pydantic import BaseModel, Field
from typing import Optional, Literal

# ----------------------------------------------------------------------
# 1. Define the Generalized Schema using Pydantic (Zero Hardcoding)
# ----------------------------------------------------------------------
class DocumentSchema(BaseModel):
    document_type: Literal[
        "Invoice", "Receipt", "Resume", "Aadhaar Card", 
        "PAN Card", "Driving License", "Passport", "Bank Statement", "Other"
    ] = Field(description="The generic category this document belongs to based on context.")
    document_title: Optional[str] = Field(None, description="The prominent heading or title found on the document.")
    corrected_text: str = Field(description="A clean, typo-corrected summary block of the core text.")
    name: Optional[str] = Field(None, description="Name of the individual, vendor, or issuing company if present.")
    date: Optional[str] = Field(None, description="Any primary date associated with the document.")
    identifying_numbers: Optional[dict] = Field(
        None, 
        description="A key-value dictionary of any unique tracking IDs found (e.g., PAN, Aadhaar, Invoice Number, Passport ID)."
    )
    contact_info: Optional[dict] = Field(
        None, 
        description="A key-value dictionary for items like Email, Phone, or Address."
    )

# ----------------------------------------------------------------------
# 2. Initialize EasyOCR Reader (Cached in Memory)
# ----------------------------------------------------------------------
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False) 

reader = load_ocr_reader()

# ----------------------------------------------------------------------
# 3. OpenCV Image Preprocessing Module
# ----------------------------------------------------------------------
def preprocess_image(cv_img):
    """Applies grayscale and denoising using OpenCV on an input image matrix."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return cv_img, denoised

# ----------------------------------------------------------------------
# 4. Generalized Validation Engine (Regex Checks)
# ----------------------------------------------------------------------
def generalized_validation(structured_dict):
    """Scans generalized property bags to run validation checks dynamically."""
    results = {}
    contacts = structured_dict.get("contact_info") or {}
    for key, value in contacts.items():
        val_str = str(value)
        if "email" in key.lower() and val_str:
            results["Email Format Check"] = "Valid" if re.match(r"[^@]+@[^@]+\.[^@]+", val_str) else "Invalid Format"
        if "phone" in key.lower() and val_str:
            clean_phone = re.sub(r"\D", "", val_str)
            results["Phone Number Format Check"] = "Valid" if len(clean_phone) >= 10 else "Invalid Format"
            
    ids = structured_dict.get("identifying_numbers") or {}
    for key, value in ids.items():
        val_str = str(value).strip().upper()
        if "pan" in key.lower() and val_str:
            results["PAN Card Format Check"] = "Valid" if re.match(r"[A-Z]{5}[0-9]{4}[A-Z]{1}", val_str) else "Invalid Format"
        if "aadhaar" in key.lower() and val_str:
            clean_aadhaar = re.sub(r"\D", "", val_str)
            results["Aadhaar Format Check"] = "Valid" if len(clean_aadhaar) == 12 else "Invalid Format"
            
    return results

# ----------------------------------------------------------------------
# 5. Local LLM Integration Modules (Ollama)
# ----------------------------------------------------------------------
def query_llm_for_structuring(raw_text):
    schema_json = json.dumps(DocumentSchema.model_json_schema(), indent=2)
    system_prompt = (
        f"You are an advanced, generalized Document AI system. Your task is to process raw OCR text, "
        f"correct any typographical mistakes, classify the document into its true high-level categories, "
        f"and extract any key metadata fields.\n\n"
        f"You MUST format your output as raw JSON that follows this JSON schema strictly:\n"
        f"{schema_json}\n\n"
        f"Do not include any chat commentary or markdown formatting blocks outside of the JSON block."
    )
    try:
        response = ollama.generate(
            model="llama3", 
            system=system_prompt,
            prompt=f"Raw Document Text:\n{raw_text}",
            options={"temperature": 0.0}
        )
        response_text = response['response'].strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()
        return json.loads(response_text)
    except Exception as e:
        return {"error": f"Failed to parse LLM structured output: {str(e)}"}

def query_llm_for_qa(raw_text, structured_json, question):
    prompt = (
        f"Context from Document:\n"
        f"--- RAW OCR ---\n{raw_text}\n\n"
        f"--- STRUCTURED DATA ---\n{json.dumps(structured_json, indent=2)}\n\n"
        f"Question: {question}\nAnswer:"
    )
    response = ollama.generate(model="llama3", prompt=prompt)
    return response['response']

# ----------------------------------------------------------------------
# 6. Streamlit App UI Layer
# ----------------------------------------------------------------------
st.set_page_config(page_title="Generalized Intelligent OCR", layout="wide")
st.title("🤖 Generalized PDF & Image Intelligent OCR Engine")
st.write("Upload an Image or a PDF. Processing happens exactly once per document asset.")

uploaded_file = st.file_uploader("Upload Document Asset...", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_id = f"proc_{uploaded_file.name}"
    
    # Check Session State Memory Cache to ensure processing runs ONLY ONCE
    if file_id not in st.session_state:
        with st.spinner("Analyzing document structure and parsing fields..."):
            
            # --- FILE CONVERSION FOR UNIFIED PROCESSING ---
            cv_images = []
            if uploaded_file.name.lower().endswith('.pdf'):
                # 🚀 Pure Python conversion using PyMuPDF (Bypasses Poppler completely)
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in doc:
                    pix = page.get_pixmap(dpi=150)
                    # Convert the raw pixel buffer directly into a NumPy matrix for OpenCV
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
                    # PyMuPDF extracts as RGB, we must convert it to BGR for OpenCV matching
                    open_cv_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    cv_images.append(open_cv_image)
                doc.close()
            else:
                # Direct Image Byte Conversion
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv_images.append(img)
            
            # --- ITERATE THROUGH PAGES FOR OCR EXTRACTION ---
            combined_ocr_text_list = []
            preprocessed_previews = []
            
            for index, cv_img in enumerate(cv_images):
                orig_img, proc_img = preprocess_image(cv_img)
                preprocessed_previews.append(orig_img) # Save reference for layout engine
                
                # Perform EasyOCR character execution
                ocr_result = reader.readtext(proc_img, detail=0)
                combined_ocr_text_list.append(f"--- PAGE {index + 1} ---\n" + "\n".join(ocr_result))
            
            raw_ocr_text = "\n\n".join(combined_ocr_text_list)
            
            # Run structured schema engine via Local LLM
            structured_data = query_llm_for_structuring(raw_ocr_text)
            
            # Run format validations
            validation_results = generalized_validation(structured_data)
            
            # Cache the complete transaction in state storage
            st.session_state[file_id] = {
                "previews": preprocessed_previews,
                "raw_ocr_text": raw_ocr_text,
                "structured_data": structured_data,
                "validation_results": validation_results
            }
            st.success("🎉 Document successfully parsed and mapped into system memory!")
            
    cached_data = st.session_state[file_id]
    
    # --- RENDER DATA LAYOUT SPLITS ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Document Preview")
        if len(cached_data["previews"]) > 1:
            page_select = st.selectbox("Select Page to View", options=range(len(cached_data["previews"])), format_func=lambda x: f"Page {x+1}")
            st.image(cached_data["previews"][page_select], use_container_width=True)
        else:
            st.image(cached_data["previews"][0], use_container_width=True)
            
    with col2:
        st.subheader("📊 Extraction Metrics")
        doc_type = cached_data["structured_data"].get("document_type", "Other")
        st.metric(label="📄 Identified Document Category", value=doc_type)
        
        st.write("**Extracted Object Properties**")
        st.json(cached_data["structured_data"])
        
        st.write("**Smart Format Validations**")
        if cached_data["validation_results"]:
            for check, status in cached_data["validation_results"].items():
                icon = "🟢" if status == "Valid" else "🔴"
                st.write(f"{icon} **{check}**: {status}")
        else:
            st.info("No active validation metrics matches triggered.")
            
    st.markdown("---")
    st.subheader("💬 Contextual Document Chat (RAG Style)")
    user_question = st.text_input("Ask a question about this document asset:")
    if user_question:
        with st.spinner("Retrieving facts..."):
            answer = query_llm_for_qa(cached_data["raw_ocr_text"], cached_data["structured_data"], user_question)
            st.write(f"**Answer:** {answer}")
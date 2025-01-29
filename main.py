import os
import shutil
import pytesseract
from pdf2image import convert_from_path
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ---------------------------------------------------------------
# STEP 1: Obtain and store baseline text
# ---------------------------------------------------------------
# Define paths
BASELINE_PDF_PATH = "data/base_document.pdf"
BASELINE_TEXT_FILE = "baseline_text.txt"


def extract_text_from_pdf(pdf_path):
    """
    Converts a PDF file into text using Tesseract OCR.
    """
    text = ""
    images = convert_from_path(pdf_path)  # Convert PDF to images

    for page_num, img in enumerate(images, start=1):
        # Perform OCR on each page
        page_text = pytesseract.image_to_string(img, lang="deu")
        text += f"--- Page {page_num} ---\n{page_text}\n"

    return text


def load_baseline_text():
    """Loads or generates baseline text from a reference PDF."""
    if not os.path.exists(BASELINE_TEXT_FILE):
        baseline_text = extract_text_from_pdf(BASELINE_PDF_PATH)
        with open(BASELINE_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(baseline_text)
    else:
        with open(BASELINE_TEXT_FILE, "r", encoding="utf-8") as f:
            baseline_text = f.read()
    return baseline_text


# ---------------------------------------------------------------
# STEP 2: Get the list of PDF documents to process
# ---------------------------------------------------------------
DOCUMENT_FOLDER = "./data/documents_to_check/augmented_documents"
SIMILARITY_THRESHOLD = 0.95  # Adjust based on testing

document_paths = [
    os.path.join(DOCUMENT_FOLDER, fname)
    for fname in os.listdir(DOCUMENT_FOLDER)
    if fname.lower().endswith(".pdf")
]


def similarity_ratio(text_a, text_b):
    """
    Computes a similarity ratio between two strings using difflib.
    Returns a float in [0.0, 1.0], where 1.0 means an exact match.
    """
    return SequenceMatcher(None, text_a, text_b).ratio()


def copy_flagged_document(doc_path, flagged_folder, ratio):
    """
    Copies a flagged document to a single timestamped folder.
    """
    os.makedirs(flagged_folder, exist_ok=True)  # Ensure folder exists

    filename = f"{ratio}_{os.path.basename(doc_path)}"  # Extract filename
    destination_path = os.path.join(flagged_folder, filename)

    shutil.copy2(doc_path, destination_path)  # Copy file with metadata
    print(f"📂 Copied flagged document to: {destination_path}")


def process_document(doc_path, baseline_text, flagged_folder):
    """
    Extracts text from a document, compares with the baseline, and copies it if flagged.
    """
    doc_text = extract_text_from_pdf(doc_path)
    ratio = similarity_ratio(baseline_text, doc_text)

    if ratio < SIMILARITY_THRESHOLD:
        copy_flagged_document(doc_path, flagged_folder, ratio)  # Copy immediately when flagged
        return doc_path, ratio  # Return only flagged documents
    return None


# ---------------------------------------------------------------
# MULTIPROCESSING: Process Documents in Parallel (Windows-Safe)
# ---------------------------------------------------------------
if __name__ == "__main__":
    baseline_text = load_baseline_text()

    # ✅ Generate timestamped folder ONCE for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flagged_folder = f"./flagged_{timestamp}"
    print(f"⚠️ Flagged documents will be copied to: {flagged_folder}")

    num_workers = min(4, os.cpu_count())  # Use up to 4 processes
    print(f"Using {num_workers} worker processes.")

    flagged_documents = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_doc = {executor.submit(process_document, doc, baseline_text, flagged_folder): doc for doc in
                         document_paths}

        for future in as_completed(future_to_doc):
            result = future.result()
            if result:  # Only add flagged documents
                flagged_documents.append(result)

    # ---------------------------------------------------------------
    # Print the Summary
    # ---------------------------------------------------------------
    print(f"\nDocuments checked: {len(document_paths)}")
    print(f"Documents flagged (similarity < {SIMILARITY_THRESHOLD}): {len(flagged_documents)}")

    for doc_path, ratio in flagged_documents:
        print(f" - {doc_path} has similarity ratio {ratio:.2f}")

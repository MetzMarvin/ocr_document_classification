import os
import shutil

import numpy as np
from pdf2image import convert_from_path
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import easyocr
import torch

# ---------------------------------------------------------------
# STEP 1: Obtain and store baseline text
# ---------------------------------------------------------------
# Define paths
BASELINE_PDF_PATH = "data/base_document.pdf"
BASELINE_TEXT_FILE = "baseline_text.txt"

DOCUMENT_FOLDER = "./data/documents_to_check/augmented_documents"

# Define Thresholds
SIMILARITY_THRESHOLD = 0.97  # Adjust based on testing
NEWLINE_SIMILARITY_THRESHOLD = 1

document_paths = [
    os.path.join(DOCUMENT_FOLDER, fname)
    for fname in os.listdir(DOCUMENT_FOLDER)
    if fname.lower().endswith(".pdf")
]


def extract_text_from_pdf(pdf_path, reader):
    """
    Converts a PDF file into text using EasyOCR.
    Each page is processed individually, and the recognized text from
    each page is concatenated into a single string.
    """
    text = ""

    # Convert PDF pages to images (PIL images)
    images = convert_from_path(pdf_path)

    # Initialize EasyOCR reader for German (adjust language list as needed)
    # GPU usage is enabled by default if available.

    for page_num, img in enumerate(images, start=1):
        # Convert the PIL image to a NumPy array.
        # pdf2image typically returns an image in RGB format.
        img_np = np.array(img)

        # Use EasyOCR's readtext; setting detail=0 returns only the recognized text.
        page_texts = reader.readtext(img_np, detail=0)

        # Join the list of text strings into one block.
        page_text = "\n".join(page_texts)
        text += f"--- Page {page_num} ---\n{page_text}\n"

    return text


def load_baseline_text(reader):
    """Loads or generates baseline text from a reference PDF."""
    if not os.path.exists(BASELINE_TEXT_FILE):
        baseline_text = extract_text_from_pdf(BASELINE_PDF_PATH, reader)
        with open(BASELINE_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(baseline_text)
    else:
        with open(BASELINE_TEXT_FILE, "r", encoding="utf-8") as f:
            baseline_text = f.read()
    return baseline_text


# ---------------------------------------------------------------
# STEP 2: Get the list of PDF documents to process
# ---------------------------------------------------------------


def similarity_ratio(text_a, text_b):
    """
    Computes a similarity ratio between two strings using difflib.
    Returns a float in [0.0, 1.0], where 1.0 means an exact match.
    """
    return SequenceMatcher(None, text_a, text_b).ratio()


def newline_count_difference(text_a, text_b):
    """
    Returns the absolute difference in the number of newline characters between two texts.
    """
    return abs(text_a.count('\n') - text_b.count('\n'))


def copy_flagged_document(doc_path, flagged_folder, ratio):
    """
    Copies a flagged document to a single timestamped folder.
    """
    os.makedirs(flagged_folder, exist_ok=True)  # Ensure folder exists

    filename = f"{ratio}_{os.path.basename(doc_path)}"  # Extract filename
    destination_path = os.path.join(flagged_folder, filename)

    shutil.copy2(doc_path, destination_path)  # Copy file with metadata
    print(f"📂 Copied flagged document to: {destination_path}")


def process_document(doc_path, baseline_text, flagged_folder, reader):
    """
    Extracts text from a document, compares with the baseline, and copies it if flagged.
    """
    print(f"Checking {doc_path}...")
    doc_text = extract_text_from_pdf(doc_path, reader)
    ratio = similarity_ratio(baseline_text, doc_text)
    newline_diff = newline_count_difference(baseline_text, doc_text)
    print(f"Checked {doc_path}. Sim-Ratio {ratio}, NEWLINE-Ratio {newline_diff}")

    if ratio < SIMILARITY_THRESHOLD or newline_diff > NEWLINE_SIMILARITY_THRESHOLD:
        copy_flagged_document(doc_path, flagged_folder, ratio)  # Copy immediately when flagged
        return doc_path, ratio  # Return only flagged documents
    return None


# ---------------------------------------------------------------
# MAIN EXECUTION (Sequential Processing)
# ---------------------------------------------------------------
if __name__ == "__main__":

    # Display CUDA information
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved()} bytes")
        torch.cuda.set_device(0)

    # Initialize EasyOCR reader for German (adjust languages as needed).
    # Ensure model_storage_directory is specified if you want to cache/download models in a specific folder.
    reader = easyocr.Reader(['de'], model_storage_directory="./models", download_enabled=True)

    # Load or generate baseline text
    baseline_text = load_baseline_text(reader)
    print("Baseline text loaded.")
    test_text = extract_text_from_pdf("data/documents_to_check/augmented_documents/aa_base_document_with_text.pdf", reader)
    print(test_text)
    print(f"SIMILARITY RATIO: {similarity_ratio(test_text, baseline_text)}")
    print(f"NEWLINE_SIMILARITY RATIO: {newline_count_difference(test_text, baseline_text)}")

    # Generate timestamped folder once for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flagged_folder = f"./flagged_{timestamp}"
    print(f"⚠️ Flagged documents will be copied to: {flagged_folder}")

    # Process each document sequentially
    flagged_documents = []
    for doc in document_paths:
        result = process_document(doc, baseline_text, flagged_folder, reader)
        if result:
            flagged_documents.append(result)

    # ---------------------------------------------------------------
    # Print the Summary
    # ---------------------------------------------------------------
    print(f"\nDocuments checked: {len(document_paths)}")
    print(f"Documents flagged (similarity < {SIMILARITY_THRESHOLD}): {len(flagged_documents)}")
    for doc_path, ratio in flagged_documents:
        print(f" - {doc_path} has similarity ratio {ratio:.4f}")

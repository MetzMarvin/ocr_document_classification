import os
import pytesseract
from pdf2image import convert_from_path
from difflib import SequenceMatcher

# ---------------------------------------------------------------
# STEP 1: Obtain and store baseline text
# ---------------------------------------------------------------
# Define paths
baseline_pdf_path = "data/base_document.pdf"
baseline_text_file = "baseline_text.txt"


def extract_text_from_pdf(pdf_path):
    """
    Converts a PDF file into text using Tesseract OCR.
    """
    text = ""
    images = convert_from_path(pdf_path)
    for page_num, img in enumerate(images, start=1):
        # Perform OCR on each page
        page_text = pytesseract.image_to_string(img)
        text += f"--- Page {page_num} ---\n{page_text}\n"
    return text


# Generate or load the baseline text
if not os.path.exists(baseline_text_file):
    baseline_text = extract_text_from_pdf(baseline_pdf_path)
    with open(baseline_text_file, "w", encoding="utf-8") as f:
        f.write(baseline_text)
else:
    with open(baseline_text_file, "r", encoding="utf-8") as f:
        baseline_text = f.read()

# ---------------------------------------------------------------
# STEP 2: OCR each incoming PDF
# ---------------------------------------------------------------
# In practice, you may have a folder containing multiple PDFs.
document_folder = "./data/documents_to_check/augmented_documents"
document_paths = [os.path.join(document_folder, fname)
                  for fname in os.listdir(document_folder)
                  if fname.lower().endswith(".pdf")]


# ---------------------------------------------------------------
# STEP 3: Compare OCR output to baseline
# ---------------------------------------------------------------
def similarity_ratio(text_a, text_b):
    """
    Computes a similarity ratio between two strings using difflib.
    Returns a float in [0.0, 1.0], where 1.0 means an exact match.
    """
    return SequenceMatcher(None, text_a, text_b).ratio()


SIMILARITY_THRESHOLD = 0.98  # Adjust this value based on experimentation

# ---------------------------------------------------------------
# STEP 4: Flag documents that differ significantly
# ---------------------------------------------------------------
flagged_documents = []

for doc_path in document_paths:
    # Extract text from the current PDF
    doc_text = extract_text_from_pdf(doc_path)
    print(f"{doc_path} -> {doc_text}")
    # Compare with baseline
    ratio = similarity_ratio(baseline_text, doc_text)
    print(f"Ratio: {ratio}")
    # Decide if the doc has extra text or significant changes
    if ratio < SIMILARITY_THRESHOLD:
        flagged_documents.append((doc_path, ratio))

# Print a summary
print("Documents checked:", len(document_paths))
print(f"Documents flagged (similarity < {SIMILARITY_THRESHOLD}): {len(flagged_documents)}\n")

for doc_info in flagged_documents:
    doc_path, ratio = doc_info
    print(f" - {doc_path} has similarity ratio {ratio:.2f}")

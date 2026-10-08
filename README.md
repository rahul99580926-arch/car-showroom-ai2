# Car Showroom AI — Photo Cleaner

This version intentionally does **not** create customer/car public links.

Workflow:
1. Upload PDF
2. Extract all photos to JPG
3. Select the authorized old-logo area once
4. Remove only, or remove + replace with your showroom logo
5. Process the full batch
6. Download all JPGs together as a ZIP

Only remove logos/watermarks you own or are authorized to remove.

Run:
```
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

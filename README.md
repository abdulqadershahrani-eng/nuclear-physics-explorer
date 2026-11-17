# AI-Powered 3D Nuclear Physics Explorer — Clean Version 5.4

This package was cleaned from the working folder provided by the project owner. It retains the existing application code and the locally saved IAEA ground-state dataset.

## Files
- `app.py` — Streamlit application
- `iaea_ground_states.csv` — locally saved IAEA LiveChart ground-state data (included; no initial download required)
- `sample_isotopes.csv` — offline demonstration fallback if the official CSV cannot be read
- `requirements.txt` — Python dependencies

## Run locally on Windows
1. Extract the ZIP and open the extracted `Nuclear_Explorer` folder.
2. Click the File Explorer address bar, type `cmd`, and press Enter.
3. Run `python -m pip install -r requirements.txt` (first time only).
4. Run `python -m streamlit run app.py`.
5. Open http://localhost:8501 if the browser does not open automatically.

Keep the Command Prompt open while using the app. Press Ctrl+C to stop it.

## Publishing online
Upload these four project files (plus this README if desired) to a GitHub repository and deploy `app.py` using Streamlit Community Cloud. The included IAEA CSV means visitors should not need to upload data. Check that the dataset is suitable for public redistribution and cite the IAEA LiveChart source. A public GitHub repository will expose the files and any contact details embedded in the application.

## Data and model caveats
The app uses simplified stable/radioactive labeling rules and a Random Forest classifier. The 3D nucleon geometry is educational, not a quantum-mechanical simulation. The locally saved IAEA data may become outdated; an optional refresh remains available in the application.

IAEA LiveChart: https://nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all

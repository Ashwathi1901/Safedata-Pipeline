# SafeData Pipeline (Streamlit Demo)

This is a working prototype of the SafeData Pipeline web app.
- Risk Assessment via k-NN linkage (simulated)
- Privacy Enhancement (SDC, DP-style noise, simple synthetic)
- Utility Measurement (stats, drift, ML sanity check)
- Reporting & Configuration (HTML/PDF, YAML)
- DPDP/GDPR checklist (non-legal)

## Miniconda Quickstart
```
cd safedata_pipeline
conda env create -f environment.yml
conda activate safedata-pipeline
streamlit run app.py
```

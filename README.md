#Element 18: Explainable GPU-Accelerated Risk Forecasting and Generative Advisory for Sustainable Aquaculture

Research paper and project report for Element 18, submitted to the NVIDIA Bootcamp 2026 (Ready to Deploy, RTD 2026 Boot Camp).

Author: Avani Guruprasad Deshpande (USN: ENG23RA0028)
Programme: B.Tech in Artificial Intelligence and Robotics, School of Engineering, Dayananda Sagar University, Bengaluru
Mentor: Dr. Godfray, Department of ECE
Academic year: 2026-27

Overview

Element 18 is a human-in-the-loop decision-support design for pond aquaculture. It forecasts the probability that a dissolved oxygen (DO) risk event will begin within the next 12 or 24 hours, explains each forecast with SHAP, retrieves approved procedure records (SOPs), and produces one validated advisory per alert event.

Main design points:

• Task: water quality risk forecasting (DO onset), not mortality prediction. Target species: Nile tilapia.
• GPU data processing with RAPIDS cuDF, compared against a pandas baseline.
• XGBoost trained with CUDA and served through the NVIDIA Triton FIL backend.
• Llama 3.1 8B served locally with Ollama, with one language model call per alert.
• Deterministic validation of generated advice, with a template fallback.
• Leakage-controlled chronological evaluation with event-level metrics.
• Fair CPU/GPU benchmark protocol and memory budget for an 8 GB laptop GPU.

Contents

File
Description
Element18_Research_Paper.pdf / .docx
IEEE-style research paper (8 pages)
Element18_Project_Report.pdf / .docx
Project report (58 pages): background, architecture, methodology, 14 algorithms, FMEA, timeline, appendices

Status

This repository contains the methodology and design. Result tables are marked as pending, and no performance, speedup or reliability result is claimed. The system is not field-validated and is not production-ready. The literature survey was not a systematic database search, and the SOP content has not been reviewed by an aquaculture specialist.

Citation

Deshpande, A. G., "Element 18: GPU-Accelerated Explainable Risk Forecasting and Generative Advisory System for Sustainable Aquaculture," Dayananda Sagar University, 2026.
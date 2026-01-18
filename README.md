# Motor Unit PIC Metrics Extension for OpenHD-EMG

This repository contains a set of scripts designed to **extend the functionality of the OpenHD-EMG framework** developed by Giacomo Valli.

The code integrates directly into the existing OpenHD-EMG analysis pipeline to **extract motor unit metrics related to persistent inward currents (PICs)** and to facilitate their visualization and export.

---

## Description

This extension allows the extraction and computation of the following motor unit discharge metrics:

- ΔF (deltaF)
- Normalized ΔF
- Brace height
- Additional associated geometric and discharge-related measures

All computed metrics are automatically **exported into structured Excel tables**, facilitating downstream statistical analysis.

In addition, the code provides dedicated functions to **visualize geometric features of the instantaneous discharge rate** of individual motor units, including plots used to illustrate brace height and related measurements.

It builds upon:
- Motor unit decomposition outputs
- Instantaneous discharge rate estimates
- Existing OpenHD-EMG data structures

To use this extension, OpenHD-EMG must be installed and properly configured.

👉 Official OpenHD-EMG repository:  
https://github.com/GiacomoValli/OpenHD-EMG

---

## Output

- Excel files containing per-motor-unit and motor-unit-pair metrics
- Optional figures illustrating geometric properties of motor unit discharge rates

---

## Intended Use

This code is primarily intended for **research applications** involving:
- Estimation of PIC-related metrics from HD-EMG data
- Motor unit discharge rate analysis

---

## Acknowledgments

This work is built upon the OpenHD-EMG framework developed and maintained by Giacomo Valli.  
Users are encouraged to cite the original OpenHD-EMG publication when using this code.



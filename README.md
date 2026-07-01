# Deep Markov Model for Neural Dynamics in Countermanding Task

This repository contains code for analyzing neurophysiological data recorded from the dorsal premotor cortex (PMd) of a macaque monkey during a **countermanding (stop-signal)** task. The goal is to understand the neural mechanisms underlying movement generation and inhibition.

A custom **Deep Markov Model (DMM)** is used to model the time-varying population activity from a 96-channel electrode array, reducing the high-dimensional data into a structured low-dimensional latent space.

## 🧠 Dataset

- ~3000 trials of neural recordings
- Each trial: 96 electrodes × 256 (reduced to 128) timepoints 
- Data collected from a macaque performing a countermanding task (GO/STOP signals)

## 🧬 Model Overview

- **DMM**: PyTorch-based recurrent variational autoencoder with Markovian temporal priors
- Learns latent dynamics of neural population activity
- Captures both **movement execution** and **inhibition** trajectories
- Supports **stochastic simulation** of realistic neural signals

## 🔍 Features

- Time series preprocessing (normalization, alignment, trial selection)
- Latent space visualization (2D/3D)
- Reconstruction accuracy and generative sampling
- Comparison between movement and inhibition conditions
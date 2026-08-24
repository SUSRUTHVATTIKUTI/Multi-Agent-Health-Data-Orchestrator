# Multi-Agent HealthTech Research System

An autonomous, multi-agent artificial intelligence system designed to retrieve, analyze, and synthesize real-time global epidemiological data using the Microsoft AutoGen framework. 

This project demonstrates the orchestration of specialized AI agents working collaboratively to solve complex analytical workflows.

## System Architecture

The system utilizes a `RoundRobinGroupChat` topology featuring three distinct AI personas:

1. **Medical Researcher (Data Ingestion Agent):** Equipped with custom asynchronous Python tools (`httpx`) to interface with live public health APIs (disease.sh). It autonomously fetches and formats raw JSON data.
2. **Clinical Analyst (Data Science Agent):** Processes the unstructured data gathered by the researcher to identify statistical trends, high-risk groups, and propose actionable AI/HealthTech interventions.
3. **Medical Director (Supervisor Agent):** Reviews the analytical output, provides a final executive assessment, and conditionally terminates the agentic loop.

## Technology Stack

* **Framework:** Microsoft AutoGen (AgentChat API v0.4+)
* **LLM:** Qwen2.5 (3B parameters) running fully locally via Ollama for privacy and zero-cost inference.
* **Networking:** `httpx` for asynchronous API calls.
* **Infrastructure:** Dockerized for consistent, isolated execution.

## Getting Started

### Prerequisites
* [Ollama](https://ollama.com/) installed and running locally.
* Docker (optional, for containerized execution).

### Local Setup

1. **Start the local LLM:**
   ```bash
   ollama pull qwen2.5:3b
   ollama run qwen2.5:3b
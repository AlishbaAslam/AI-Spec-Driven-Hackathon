<!-- Sync Impact Report:
     Version change: 0.1.0 → 1.0.0
     Added sections: All principles and sections based on user requirements
     Templates requiring updates: N/A (initial constitution)
     Follow-up TODOs: None
-->
# Unified Book on Physical AI and Humanoid Robotics with Integrated RAG Chatbot Constitution

## Core Principles

### Accuracy through verification of technical concepts and tools
All technical claims and examples must be traceable to official documentation or sources; Citation format: Markdown with hyperlinks for online resources; Source types: minimum 50% official documentation from tools like ROS 2, NVIDIA Isaac, OpenAI, etc.

### Clarity for educational audience (students with AI background)
Writing clarity: Flesch-Kincaid grade 8-10 for accessibility; Content must be understandable for students with AI background

### Reproducibility (all code, simulations, and deployments documented and executable)
All code examples, simulations, and deployments must be documented and executable; All content must be reproducible by users

### Rigor (use of industry-standard tools and best practices in robotics and AI)
Use of industry-standard tools and best practices in robotics and AI; All implementations must follow established standards

### Plagiarism check: 0% tolerance; all content original or properly attributed
0% tolerance for plagiarism; all content must be original or properly attributed; All content must be traceable to sources

### Integrated RAG Chatbot Functionality
RAG chatbot must be embedded in the book, capable of answering questions on book content and user-selected text; Chatbot integration with Docusaurus, FastAPI, Neon Serverless Postgres, Qdrant Cloud Free Tier

## Technical Standards and Tooling
Tools: Must use Docusaurus for book creation, GitHub Pages for deployment, Spec-Kit Plus, Claude Code, OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres, Qdrant Cloud Free Tier for RAG chatbot; Content focus: AI Systems in the Physical World, Embodied Intelligence; bridging digital AI to physical robotics

## Book Structure and Content Requirements
Structure: Follow quarter overview with 4 modules, plus capstone project section; All modules covered with practical examples, simulations, and code snippets; Passes functionality tests: chatbot handles voice commands, path planning, and object manipulation in simulations

## Governance
All outputs strictly follow the user intent; Prompt History Records (PHRs) are created automatically and accurately for every user prompt; Architectural Decision Record (ADR) suggestions are made intelligently for significant decisions; All changes are small, testable, and reference code precisely

**Version**: 1.0.0 | **Ratified**: 2025-12-10 | **Last Amended**: 2025-12-10

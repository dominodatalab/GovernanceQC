# Governance Explorer App

Governance Explorer is a Streamlit-based dashboard for exploring and visualizing governance bundles, their audit history, and key metrics.  It is designed to connect to a Domino Governance API (or run in a Domino App) and provides an interactive, visual interface for navigating compliance workflows.

## Features

The app currently contains **three main tabs**:

### 1. Bundles  
- Displays a list of all governance bundles returned by the API.  
- Each row includes:
  - **Bundle Name**
  - **Project Name**
  - **QC Plan / Policy Name**
  - **Current Stage**
  - **Current Stage Assignee**
  - **Created Date**
- The table is fully scrollable, sortable, and adapts to available screen width.

<img width="2880" height="1410" alt="image" src="https://github.com/user-attachments/assets/4ea4baf6-c7bd-4a42-a4c5-799457212064" />



### 2. History  
- Allows selection of a specific bundle to view its **audit trail**.
- **Stage Change Timeline**:
  - A visual, horizontal workflow diagram showing each stage the deliverable has passed through.
  - Highlights the **current stage** in green.
  - Marks **pending** stages in grey.
  - Marks **reverted** stages in red with a dashed arrow and rejection note.
  - Shows dates and times of each transition.
- **Audit Log Table**:
  - Lists timestamp, user, event type, project, change type, before/after values, and notes for each audit event.

    <img width="2880" height="3088" alt="image" src="https://github.com/user-attachments/assets/852529ef-53bf-4962-81dc-264a23ebddc2" />


### 3. Metrics  
- Displays high-level metrics for all bundles:
  - **Bundles by Current Stage** (bar chart).
  - **Days Since Created** for each bundle in its current stage.
- Helps identify bottlenecks and track throughput.

<img width="2880" height="1410" alt="image" src="https://github.com/user-attachments/assets/ccd23903-00fd-4de1-9f8b-627382338134" />


## Technology Stack
- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Data Fetching**: Python `requests` library for calling the Domino Governance API.
- **Visualization**: HTML + SVG components for the stage-change timeline, Streamlit native charts for metrics.

## Prerequisites
- **Python Version**: Python 3.9+ recommended
- **Environment Access**:  
  Access to a Domino environment with:
  - **Domino Governance** enabled
  - **Domino Audit Trail API** enabled
- **Required Python Packages**:
  | Package | Purpose |
  |---------|---------|
  | **streamlit** | Main framework for building the interactive dashboard UI |
  | **pandas** | Handling tabular data for bundle lists, audit logs, and metrics |
  | **requests** | Making API calls to the Domino Governance API |
  | **python-dateutil** *(optional)* | Robust datetime parsing |

---

## License

This project is released under the **MIT License**.  
You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the Software.



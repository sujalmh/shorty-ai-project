# AI Excel Spreadsheet Application

## Overview

[![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white&style=flat-square)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/-React-61DAFB?logo=react&logoColor=white&style=flat-square)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/-TypeScript-3178C6?logo=typescript&logoColor=white&style=flat-square)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/-TailwindCSS-06B6D4?logo=tailwindcss&logoColor=white&style=flat-square)](https://tailwindcss.com/)
[![Vite](https://img.shields.io/badge/-Vite-646CFF?logo=vite&logoColor=white&style=flat-square)](https://vitejs.dev/)
[![AG Grid](https://img.shields.io/badge/AG--Grid-000000?logo=ag--grid&logoColor=white&style=flat-square)](https://www.ag-grid.com/)
[![Plotly](https://img.shields.io/badge/-Plotly-326FEA?logo=plotly&logoColor=white&style=flat-square)](https://plotly.com/javascript/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is an AI-powered spreadsheet application designed to provide an interactive and intuitive user experience for data analysis and manipulation. It features a conversational AI interface that allows users to perform complex operations on their data using natural language, along with a robust data grid and charting capabilities.

## Screenshots
<!-- Add your actual screenshots here -->
A visual demonstration of the AI Excel application in action.

**Main Interface**
<img width="1755" height="902" alt="Main Interface Screenshot" src="https://github.com/user-attachments/assets/2b7fedbe-8b6b-42bf-900c-0ab80141e560" />
<img width="1919" height="767" alt="image" src="https://github.com/user-attachments/assets/f19295fc-b23b-4b66-9741-7fa70e3d90d8" />

**Diagrams/Plots**
<img width="1755" height="902" alt="Diagrams/Plots Screenshot" src="https://github.com/user-attachments/assets/f4105f94-07c0-4770-ba56-66196d91ddbe" />
<img width="1906" height="982" alt="Diagrams/Plots Screenshot" src="https://github.com/user-attachments/assets/ea5d286f-65d2-472c-b901-11cef7f24935" />


## Features

### Core Functionality
- **AI-Powered Chat Interface**: Interact with your data using natural language commands (e.g., "add a column for profit = revenue - cost", "highlight rows where profit < 0").
- **Dynamic Spreadsheet Grid**: Built with AG Grid, providing a feature-rich, high-performance data table.
- **Data Import/Export**: Easily import CSV and Excel files.
- **Computed Columns**: Create new columns based on expressions.
- **Row & Column Management**: Add new rows and empty columns.
- **Interactive Plotting**: Generate various charts and diagrams based on data insights, integrated with Plotly.js.
- **Real-time Updates**: Changes made through the AI or manual edits are reflected instantly.

### Design System (Frontend)
The frontend boasts a clean, professional, and cohesive design system with a modern glass morphism aesthetic.

**Key Design Principles:**
- **Consistency**: Unified styling across all components.
- **Visual Hierarchy**: Clear distinction between primary, secondary, and tertiary elements.
- **Accessibility**: WCAG AA compliant contrast ratios and clear focus states.
- **Smooth Interactions**: Subtle transitions and tactile feedback.

**Design Tokens:**
- **Color Palette**: Primary Indigo (600-700), Secondary Slate, Accent Blue, Semantic colors (Success, Warning, Error).
- **Spacing System**: Standardized gaps and padding for predictable layouts.
- **Typography**: Optimized font sizes and weights for readability.
- **Border Radius**: Consistent rounding for a modern look.

**Component Standardization:**
- **Buttons**: 5 variants (Primary, Secondary, Tertiary, Ghost, Danger) with consistent sizing, padding, and hover/active states.
- **Input Fields**: Light and dark themes for consistent user input.
- **Tabs**: Clear active/inactive states with badge support.
- **Chat Bubbles**: Distinct styling for user and assistant messages.

**Visual Enhancements:**
- **Glass Morphism**: Dark effect for the chat panel, light effect for the sheet/diagrams area, providing depth and elegance.
- **Custom Scrollbars**: Improved aesthetics for scrollable areas.
- **AG Grid Theming**: Fully customized to integrate seamlessly with the application's visual language, including intelligent row highlighting.

## Application Architecture

```mermaid
C4Context
    title System Context for AI Excel Application

    Person(user, "User", "Uses the AI Excel application to analyze and manipulate data.")

    System(frontend, "Frontend Application", "React/TypeScript application with AG Grid and Plotly for interactive data display and charting. Implements a modern glass morphism design system.")
    System(backend, "Backend Services", "Python-based API for data management, spreadsheet operations, and AI integration.")
    System_Ext(ai_service, "AI Service / LLM", "External or internal service providing natural language processing and action inference capabilities.")

    Rel(user, frontend, "Interacts with", "Uses Web Browser")
    Rel(frontend, backend, "Communicates with", "REST API (Axios)")
    Rel(backend, ai_service, "Delegates AI tasks to", "API Calls")
    Rel(backend, frontend, "Sends data and AI responses to", "JSON via REST API")
```

## Technologies Used

### Frontend
- **React**: A JavaScript library for building user interfaces.
- **TypeScript**: Statically typed superset of JavaScript.
- **Vite**: Fast development build tool.
- **Tailwind CSS**: A utility-first CSS framework for rapid UI development.
- **AG Grid**: Advanced data grid for React.
- **Plotly.js**: JavaScript graphing library for interactive charts.
- **Axios**: Promise-based HTTP client.

### Backend
- **Python**: Core programming language for backend services.
- **FastAPI**: Modern, fast (high-performance) web framework for building APIs with Python.
- Data processing libraries (e.g., Pandas for spreadsheet operations)
- AI/ML libraries (e.g., for natural language processing and action inference)

## Getting Started

### Prerequisites

- Node.js (v18+) and npm
- Python (v3.8+) (for backend, if applicable)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [repository_url]
   cd ai-excel
   ```

2. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   ```

3. **Backend Setup:**
   *(Instructions depend on backend implementation. Assuming a Python backend for now.)*
   ```bash
   cd ../backend
   pip install -r requirements.txt # (assuming a requirements.txt exists)
   # Or install dependencies manually if known
   ```

### Running the Application

1. **Start the Backend Server:**
   *(Example for a common Python setup)*
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Start the Frontend Development Server:**
   Open a new terminal.
   ```bash
   cd frontend
   npm run dev
   ```
   The frontend application will typically open in your browser at `http://localhost:5174` (or another available port).

## Project Structure

```
.
├── backend/                  # Backend services and API
│   ├── app/
│   │   ├── main.py           # Main backend application
│   │   ├── models/           # Data models (schemas.py)
│   │   ├── routes/           # API endpoints (api.py)
│   │   └── services/         # Business logic (ai_service.py, data_service.py)
│   └── ...
├── frontend/                 # Frontend React application
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── App.tsx           # Main application component
│   │   ├── index.css         # Global styles and design system
│   │   ├── main.tsx          # Entry point
│   │   ├── components/       # Reusable UI components
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── SheetGrid.tsx
│   │   │   ├── DiagramGallery.tsx
│   │   │   └── PlotViewer.tsx
│   │   ├── lib/              # API clients and utilities
│   │   └── types/            # TypeScript type definitions
│   ├── package.json          # Frontend dependencies and scripts
│   └── tailwind.config.js    # Tailwind CSS configuration
├── scripts/                  # Utility scripts
├── DESIGN_SYSTEM.md          # Detailed documentation of the design system
└── README.md                 # Project README
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes.
4. Ensure tests pass and code adheres to style guidelines.
5. Submit a pull request.

## License

This project is licensed under the MIT License.

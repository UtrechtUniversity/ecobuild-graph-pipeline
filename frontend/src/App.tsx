import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ExperimentList from './components/ExperimentList';
import ExperimentDetail from './components/ExperimentDetail';
import AddExperimentRun from './components/AddExperimentRun';
import './App.css'; // We'll create this

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Experiment Dashboard</h1>
          <nav>
            <Link to="/">Home</Link> {/* Add more links as needed */}
            <Link to="/add-run">Add Experiment Run</Link>
          </nav>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<ExperimentList />} />
            <Route path="/experiments/:id" element={<ExperimentDetail />} />
            <Route path="/add-run" element={<AddExperimentRun />} />
            {/* Add more routes if needed */}
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

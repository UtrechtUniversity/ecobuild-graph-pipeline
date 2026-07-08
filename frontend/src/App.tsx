import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { FlaskConical } from 'lucide-react';
import ExperimentList from './components/ExperimentList';
import ExperimentDetail from './components/ExperimentDetail';
import PaperList from './components/PaperList';
import PaperReview from './components/PaperReview';
import ServiceStatus from './components/ServiceStatus';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card">
          <div className="mx-auto flex max-w-5xl items-center gap-3 px-6 py-4">
            <FlaskConical className="size-6 text-primary" />
            <h1 className="text-xl font-semibold tracking-tight">
              <Link to="/">EcoBuild Paper Knowledge Extractor</Link>
            </h1>
            <ServiceStatus />
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">
          <Routes>
            <Route
              path="/"
              element={
                <div className="flex flex-col gap-8">
                  <PaperList />
                  <ExperimentList />
                </div>
              }
            />
            <Route path="/experiments/:id" element={<ExperimentDetail />} />
            <Route path="/papers/:id/review" element={<PaperReview />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

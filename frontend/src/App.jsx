import React, { useState } from 'react';
import './App.css';

function App() {
  const [userId, setUserId] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!userId.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    // Normalize user_id to match format USR_XXXXX
    let queryId = userId.trim().toUpperCase();
    if (queryId.startsWith('USR_') && queryId.length < 9) {
      const numPart = queryId.substring(4);
      if (!isNaN(numPart)) {
        queryId = `USR_${numPart.padStart(5, '0')}`;
      }
    } else if (!isNaN(queryId)) {
      queryId = `USR_${queryId.padStart(5, '0')}`;
    }

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/predict-churn/${queryId}`);
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`User ID "${queryId}" not found in database.`);
        }
        throw new Error('Server returned an error. Make sure your FastAPI backend is running.');
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    if (risk >= 0.70) return '#ef4444'; // Red
    if (risk >= 0.35) return '#f97316'; // Orange
    return '#22c55e'; // Green
  };

  const getRiskBackground = (status) => {
    if (status === 'High Risk') return 'status-high';
    if (status === 'Medium Risk') return 'status-medium';
    return 'status-low';
  };

  const getRecommendation = (risk) => {
    if (risk >= 0.70) {
      return {
        action: 'Trigger immediate loyalty discount offer and schedule customer success manager callback.',
        badge: 'Critical Action Needed'
      };
    }
    if (risk >= 0.35) {
      return {
        action: 'Email targeted discount coupon and request customer feedback survey.',
        badge: 'Recommended Action'
      };
    }
    return {
      action: 'Account is healthy. Keep standard engagement policies.',
      badge: 'Account Healthy'
    };
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-title">
          <span className="logo-icon">🔮</span>
          <h1>Multimodal Churn Predictor</h1>
        </div>
        <p className="subtitle">Real-time Customer Retention Intelligence</p>
      </header>

      <main className="main-content">
        <section className="search-section">
          <form onSubmit={handleSearch} className="search-form">
            <input
              type="text"
              placeholder="Enter User ID (e.g. USR_00001, USR_00003 or 1)"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="search-input"
              disabled={loading}
            />
            <button type="submit" className="search-button" disabled={loading}>
              {loading ? 'Analyzing...' : 'Analyze Risk'}
            </button>
          </form>
          <div className="helper-text">
            <span>Tip: Try <strong>USR_00001</strong> for high risk, or <strong>USR_00003</strong> for low risk.</span>
          </div>
        </section>

        {error && (
          <div className="error-alert">
            <span className="alert-icon">⚠️</span>
            <div className="alert-content">
              <strong>Prediction Failed</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {loading && (
          <div className="loader-container">
            <div className="spinner"></div>
            <p>Evaluating multi-modal features...</p>
          </div>
        )}

        {result && (
          <div className={`dashboard-grid ${getRiskBackground(result.status)}`}>
            {/* Risk Gauge Panel */}
            <div className="dashboard-card card-hero">
              <h2>Churn Probability</h2>
              <div className="risk-gauge-container">
                <svg className="risk-gauge" viewBox="0 0 100 100">
                  <circle className="gauge-bg" cx="50" cy="50" r="40" />
                  <circle
                    className="gauge-progress"
                    cx="50"
                    cy="50"
                    r="40"
                    style={{
                      stroke: getRiskColor(result.churn_risk),
                      strokeDasharray: `${result.churn_risk * 251.2} 251.2`
                    }}
                  />
                  <text className="gauge-text" x="50" y="55" fill={getRiskColor(result.churn_risk)}>
                    {Math.round(result.churn_risk * 100)}%
                  </text>
                </svg>
              </div>
              <div className="status-label">
                <span className={`badge ${result.status.toLowerCase().replace(' ', '-')}`}>
                  {result.status}
                </span>
              </div>
            </div>

            {/* Recommendations Panel */}
            <div className="dashboard-card card-recommendation">
              <div className="card-header">
                <h2>Retention Playbook</h2>
                <span className="rec-badge">{getRecommendation(result.churn_risk).badge}</span>
              </div>
              <div className="recommendation-content">
                <div className="playbook-item">
                  <strong>Triggering Event:</strong>
                  <p className="trigger-text">{result.trigger}</p>
                </div>
                <div className="playbook-item action-plan">
                  <strong>Recommended CS Action:</strong>
                  <p>{getRecommendation(result.churn_risk).action}</p>
                </div>
              </div>
            </div>

            {/* Tabular Telemetry Panel */}
            <div className="dashboard-card card-telemetry">
              <h2>Operational Activity Logs</h2>
              <div className="metrics-grid">
                <div className="metric-item">
                  <span className="metric-label">Logins (30 Days)</span>
                  <span className="metric-value">{result.login_count_30d}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Total Spend</span>
                  <span className="metric-value">${result.total_spend.toFixed(2)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Days Inactive</span>
                  <span className="metric-value">{result.days_since_last_login} days</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Subscription Tier</span>
                  <span className="metric-value tier-badge">{result.subscription_type}</span>
                </div>
              </div>
            </div>

            {/* Support Ticket NLP Panel */}
            <div className="dashboard-card card-nlp">
              <h2>Support Ticket & Sentiment</h2>
              <div className="ticket-nlp-content">
                <div className="ticket-body-container">
                  <strong>Recent Support Message:</strong>
                  {result.tickets && result.tickets.length > 0 ? (
                    <blockquote className="ticket-quote">
                      "{result.tickets[0]}"
                    </blockquote>
                  ) : (
                    <p className="no-tickets">No recent support tickets found.</p>
                  )}
                </div>
                
                <div className="sentiment-analysis-panel">
                  <div className="sentiment-header">
                    <strong>Evaluated Sentiment Score:</strong>
                    <span className="sentiment-value" style={{ color: getRiskColor(1 - result.support_sentiment_score) }}>
                      {result.support_sentiment_score.toFixed(4)}
                    </span>
                  </div>
                  <div className="sentiment-bar-container">
                    <div className="sentiment-bar-labels">
                      <span>Angry (0.0)</span>
                      <span>Happy (1.0)</span>
                    </div>
                    <div className="sentiment-bar-track">
                      <div
                        className="sentiment-bar-fill"
                        style={{
                          width: `${result.support_sentiment_score * 100}%`,
                          backgroundColor: getRiskColor(1 - result.support_sentiment_score)
                        }}
                      />
                    </div>
                  </div>
                  <p className="sentiment-hint">
                    Scores closer to 0.0 denote customer frustration, triggering classification alarms.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
